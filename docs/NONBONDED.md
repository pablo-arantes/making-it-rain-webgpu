# The nonbonded decomposition

What OpenMM puts in which force group, and the exact form of every term. All of
it measured rather than inferred, because the obvious reading is wrong in three
separate places and each one is worth more than the total energy.

`tools/build_nonbonded_fixture.py` is the measurement. It builds a small system
by hand, decomposes the energy independently in numpy, and prints both
alongside OpenMM's own two groups.

## The split

OpenMM's `NonbondedForce` reports into two force groups: its own, and
`recipForceGroup`. The division is not pair-sum versus mesh-sum.

**Direct-space group:**

| term | over | form |
| --- | --- | --- |
| screened Coulomb | non-excluded pairs within the cutoff | `ONE_4PI_EPS0 · qᵢqⱼ · erfc(αr) / r` |
| Lennard-Jones | non-excluded pairs within the cutoff | `4ε((σ/r)¹² − (σ/r)⁶)` |
| **exclusion correction** | **every exception, any distance** | **`−ONE_4PI_EPS0 · qᵢqⱼ · erf(αr) / r`** |
| exception interaction | every exception, any distance | `ONE_4PI_EPS0 · chargeProd / r` plus LJ with the exception's own σ, ε |
| dispersion correction | — | analytic, see below |

**Reciprocal-space group:**

| term | form |
| --- | --- |
| mesh sum | smooth PME, order-5 B-splines |
| self energy | `−ONE_4PI_EPS0 · α · Σqᵢ² / √π` |
| **net-charge correction** | **`−π Q² · ONE_4PI_EPS0 / (2α²V)`** |

Combination rules are the arithmetic mean for σ and the geometric mean for ε.
α comes from `sqrt(−log(2·tol))/cutoff`, which for the notebook's 1.0 nm cutoff
and `ewaldErrorTolerance = 0.0005` is 2.628260884878466 nm⁻¹ for every system.

Verified: the direct-space hypothesis reproduces OpenMM's group to **2e-9
kJ/mol**, and the reciprocal one to **2e-7 relative**.

## The three things that are not obvious

### The exclusion correction is most of the direct group

Reciprocal space sums over *every* pair, including the ones the force field
excludes. Each excluded pair's reciprocal contribution therefore has to be taken
back out in real space, using the particles' **own charges** rather than the
exception's `chargeProd`.

These are not small. For the solvated protein–ligand fixture:

| term | kJ/mol |
| --- | ---: |
| screened Coulomb over pairs | small and negative |
| **exclusion corrections** | **+1.67e6** |
| direct-space group total | +1.56e6 |
| reciprocal-space group total | −1.73e6 |
| **system total** | **−1.38e5** |

A direct-space kernel written as a plain pair sum would be wrong by more than
ten times the answer, and it would look plausible while doing it — the sign and
rough magnitude of a water box's pair energy are easy to talk yourself into.

Why so large and so positive: for a TIP3P water the O–H exclusion correction is
about +141 kJ/mol each and the H–H one about −68, so roughly +214 per water, and
there are 7800 of them.

### The dispersion correction averages over pairs *including* i = j

```
E = 8π N² / V · ( ⟨εσ¹²⟩ / (9 rc⁹) − ⟨εσ⁶⟩ / (3 rc³) )
```

The averages run over all unordered pairs **including each particle with
itself**, so N(N+1)/2 of them, with the combination rules applied to each pair.
Determined by toggling `setUseDispersionCorrection` and fitting: it matches to
nine decimal places. It depends on the volume but on no coordinate, so it
contributes energy and pressure but no force.

Implemented grouped by distinct (σ, ε) rather than looped over pairs, because
N(N+1)/2 is 3.6e8 for the solvated fixture; the grouped and brute-force versions
agree to 1e-11, which is f64 summation order.

### A non-neutral system needs the background term

```
E = −π Q² · ONE_4PI_EPS0 / (2α²V)
```

Ewald summation implicitly immerses the cell in a neutralising background, and
this is that background's self-interaction. It was found by a 0.32 kJ/mol
disagreement on a fixture carrying a deliberate net charge of −0.416 e, and
confirmed by sweeping the net charge from 0 to 2 e.

`addSolvent` neutralises, so for anything the app builds this is near zero — but
not exactly zero once a ligand's charges have been rounded, and a user can
upload any system they like. No coordinate dependence, so no force.

## What this cost to find

The decomposition came out of comparing an independent numpy implementation
against OpenMM term by term, on a system small enough to also sum in closed
form. Three things to keep in mind if this ever needs revisiting:

* **Use the `Ewald` method, not `PME`, to pin the physics.** Ewald is an exact
  k-space sum, so a brute-force reference can match it to machine precision;
  PME differs from it by the mesh error and would have hidden a real bug of the
  same size. The committed fixture ships as PME, because that is what the
  notebook runs, but it records both.
* **Give the fixture a net charge and a particle with no LJ at all.** Both
  found bugs. A neutral, fully-parameterised system would have passed with the
  net-charge term missing.
* **Grid convergence separates "my PME is wrong" from "my formula is wrong".**
  Raising the mesh from 16³ to 48³ made our reciprocal sum converge onto the
  exact Ewald sum to 7e-6, which proved the mesh code right and pointed at a
  missing term rather than a coding error.

## Current agreement

Reference implementations in `src/reference/forces/`, kernels in
`src/gpu/kernels/`, both against OpenMM's Reference platform.

| | energy vs OpenMM | force rel. RMS vs our f64 |
| --- | ---: | ---: |
| direct space, f64 reference, 1L2Y and 3HTB | < 1e-10 relative | — |
| reciprocal space, f64 reference, 1L2Y | < 1e-9 relative | — |
| **direct space, WGSL, 1L2Y** | **1.05e-7** | **8.38e-7** |
| **direct space, WGSL, 3HTB (26,749 atoms)** | **3.93e-7** | **1.13e-5** |

The WGSL direct-space force error is about ten times the bonded terms', from
two causes: the kernel's erfc is a rational approximation good to 1.2e-7
relative rather than an exact function, and the terms cancel hard enough that a
relative error in the large ones shows up magnified in the sum.

Reciprocal space on the GPU is not written yet; `docs/HANDOFF.md` has what is
left.
