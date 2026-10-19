# Dynamics: constraints and integrators

What OpenMM's integrator and constraint solver actually do, and the four places
where the obvious reading is wrong. Measured by
`tools/capture_constraints.py`, which captures OpenMM's solver and two
deterministic trajectories as oracles.

## The integrator

`LangevinIntegrator` is `LangevinMiddleIntegrator` in OpenMM ≥ 8.2 (see
[NOTEBOOK_PARITY.md](NOTEBOOK_PARITY.md) finding 6), so the notebook runs
LFMiddle. From `ReferenceLangevinMiddleDynamics::update`, verbatim in structure:

```
v += dt · F/m                        full kick, not a half kick
applyToVelocities(x, v, tol)         RATTLE
xP  = x + v·(dt/2)
v   = vscale·v + noisescale·√(kT/m)·ξ
xP += v·(dt/2)
oldx = xP                            the UNCONSTRAINED trial positions
apply(x, xP, tol)                    SHAKE, reference = x
v += (xP − oldx)/dt                  the constraint's effect on the velocity
x  = xP
```

with `vscale = exp(−dt·friction)` and `noisescale = √(1 − vscale²)`.

Three things to notice. The kick is a **full** `dt`, which is what makes this
LFMiddle rather than velocity Verlet. Velocities are RATTLE-projected
immediately after the kick, and **not** again at the end of the step. And
`oldx` holds the trial positions *before* SHAKE, so `(xP − oldx)/dt` is the
velocity the constraint projection implies.

### 🔴 Kinetic energy is reported differently per integrator, and that is the whole story

**Corrected.** An earlier version of this file said LFMiddle at zero friction
is "symplectic Euler, not velocity Verlet" and does not conserve energy. That
was wrong, and the real explanation matters more.

`VerletIntegrator` is, verbatim from `ReferenceVerletDynamics::update`:

```
v += dt · F/m
xP = x + dt · v
apply(x, xP, tol)                    SHAKE
v  = (xP − x)/dt
x  = xP
```

Set `friction = 0` in LFMiddle and the two are the *same dynamics*. The two
half drifts collapse into one full drift, and `v += (xP − oldx)/dt` is
algebraically `v = (xP − x)/dt`, since `oldx = x + dt·v_kicked`. The only
difference is that LFMiddle also RATTLE-projects the velocities right after the
kick.

What differs is **how the kinetic energy is reported**. From
`ReferenceKernels.cpp`:

| integrator | kinetic energy from |
| --- | --- |
| `VerletIntegrator` | velocities shifted by **+dt/2**, then RATTLE'd at a 1e-4 tolerance |
| `LangevinIntegrator`, `LangevinMiddleIntegrator` | the velocities as they are |

and `computeShiftedVelocities` is

```cpp
velocities[i] = velData[i] + forceData[i]*(timeShift/masses[i]);
if (timeShift != 0)
    constraints.applyToVelocities(posData, velocities, inverseMasses, 1e-4);
```

The shift is the standard leapfrog correction: a leapfrog integrator carries its
velocities half a step out of step with its positions, so a kinetic energy
computed from them is evaluated at the wrong time. Measured on 1L2Y,
equilibrated, over 40 steps at 2 fs — the range of the total energy, per atom:

| | Verlet trajectory | LFMiddle trajectory |
| --- | ---: | ---: |
| PE + KE as OpenMM reports it | **3.08e-3** | 2.54e-2 |
| PE + KE from unshifted velocities | 2.54e-2 | 2.54e-2 |
| PE + KE from `v + dt/2·F/m`, **no** RATTLE | 2.06e-2 | 2.06e-2 |

So the conserved quantity is the shifted *and re-constrained* one, and it is
about eight times better than the unshifted figure. The dynamics was never the
problem. Skipping the RATTLE gets most of the way to nothing: the shift moves
velocities off the constraint surface, and the components along the constrained
bonds then contribute kinetic energy that should not exist.

**Two consequences.**

*For the NVE drift gate:* it has to use the shifted, re-constrained kinetic
energy, whichever integrator produced the trajectory. Running
`VerletIntegrator` is the easy way to get that, because OpenMM applies the shift
for you — which is what `tools/calibrate_thresholds.py` does, and why the
threshold it measured is 0.17 kJ/mol/ns/atom.

*For parity, which matters more:* the notebook's `StateDataReporter` logs
`kineticEnergy`, `totalEnergy` and `temperature` every 10 ps, and the notebook
uses `LangevinIntegrator`. So **the engine's reporters must use the unshifted
convention**, or every line of every log file disagrees with the notebook's. The
convention is a property of the integrator, not a free choice, and
`src/reference/kinetic.ts` implements both.

## The constraint solver

From `ReferenceCCMAAlgorithm::applyConstraints`, with the CCMA preconditioner
absent — which is plain SHAKE, solving the same system and reaching the same
fixed point:

```
r_ij[c]  = x[i] − x[j]              from the reference positions, fixed
d2[c]    = |r_ij[c]|²
reduced[c] = 0.5 / (1/m_i + 1/m_j)

repeat:
  for every constraint c:                      ← all deltas, from the same trial
    positions:   δ = reduced·(d² − |rp|²) / (rp · r_ij)
                 converged when |rp|² ∈ [(1−t)², (1+t)²]·d²
    velocities:  δ = −2·reduced·(vp · r_ij) / d2
                 converged when |δ| ≤ t
  if all converged: stop
  for every constraint c:                      ← then all applied
    dr = r_ij[c]·δ[c];  xP[i] += dr/m_i;  xP[j] −= dr/m_j
```

**The deltas are applied simultaneously** — Jacobi, not Gauss-Seidel. That is
what makes a per-cluster GPU kernel an exact match rather than an
approximation, since constraints in different clusters never interact. And the
correction direction is the **reference** bond vector, held fixed through the
iteration.

The tolerance is relative, as OpenMM's documentation says: a fraction of the
constrained distance.

### 🔴 `applyConstraints` is its own reference

`Context.applyConstraints(tol)` passes the **same array** as reference and
trial: the correction directions are the current bond vectors. That is not what
SHAKE does inside an integrator, where the reference is the previous step's
positions.

Getting this wrong put our answer 0.04 nm from OpenMM's while every constraint
was still satisfied to 1e-6 — so the solver looked correct and the comparison
looked broken. With the reference set the way `applyConstraints` sets it, we
agree with OpenMM to **1.8e-7 nm** on 1L2Y and **6.2e-7 nm** on 3HTB, which is
the width of the tolerance band and therefore the floor for the comparison.

### 🔴 A freshly built system violates its own constraints

Measured on both fixtures: the worst constrained distance is **25% away** from
the force field's value, and SHAKE moves atoms up to 0.02 nm to fix it.

`Modeller.addHydrogens` places hydrogens geometrically and then relaxes them
for only 50 iterations at a 1.0 kJ/mol/nm tolerance
([NOTEBOOK_PARITY.md](NOTEBOOK_PARITY.md) finding 2), which is nowhere near
tight enough to put an X–H bond at the force field's equilibrium length.

So dynamics cannot simply start from a built system. The notebook gets the
projection implicitly, from `minimizeEnergy`; the engine has to do it
explicitly, and the run record should say it did.

### Plain SHAKE is enough, at the step sizes that matter

Iteration counts to reach a 1e-6 relative tolerance:

| displacement | 1L2Y | 3HTB |
| --- | ---: | ---: |
| 0.01 nm, the stress capture | 89 | **150 (the cap)** |
| 1e-4 nm, the scale of a 2 fs step | < 40 | < 40 |

So plain SHAKE hits OpenMM's own 150-iteration ceiling on a displacement a
hundred times larger than a real step, and converges comfortably at a real one.
CCMA's refinement — premultiplying the deltas by an approximate inverse of the
constraint coupling matrix — is the fix if that ever becomes the bottleneck, and
for the clusters this workload produces there is very little coupling to
accelerate anyway.

### 🔴 SETTLE is not an optimisation. It changes the answer.

OpenMM does not use one constraint algorithm. From
`ReferenceConstraints::ReferenceConstraints`, it finds every **closed triangle
of three atoms where each atom holds exactly two constraints** and gives those
to `ReferenceSETTLEAlgorithm`; everything else goes to CCMA.

That triangle is rigid water, and it is **88 to 90% of all clusters** — 688 of
784 in 1L2Y, 8021 of 8861 in 3HTB. The central atom is the one whose two bonds
are equal, and 🔴 **OpenMM compares those two lengths after rounding them to
`float`**, so bonds differing in the ninth decimal are equal to it and must be
here too, or a cluster lands in a different solver.

SETTLE is analytic: it computes the rigid-body placement that satisfies all
three distances exactly, so its answer is good to machine precision rather than
to a tolerance. With it in place our water clusters come out at **3.085e-8**
relative constraint error — the same figure OpenMM's own frames show, far inside
its stated 1e-6 — against 9.9e-7 for SHAKE. That is what an analytic solution
looks like, and it is how to tell the two apart.

So `docs/PLAN.md`'s "SETTLE for water, CCMA-style for bonds to H" is a parity
requirement, not the performance note it reads as.

### 🔴 And a step is not just the integrator

`CMMotionRemover` is in every `System` these force fields build, with
`frequency = 1`, so it runs **every step**. It is not a force: OpenMM applies it
from `ContextImpl::updateContextState`, which the integrator calls at the start
of a step, *before* the forces are computed, and it subtracts the mass-weighted
mean velocity from every particle.

Leaving it out hides well. It shifts every atom by the same amount, so every
constrained distance stays satisfied and every constraint test still passes —
the trajectory just slides, by `|v_com| · dt`, which was 2.2e-5 nm on 1L2Y.

### What it took to reproduce a step, and a lesson about residuals

One 2 fs step, worst position difference from OpenMM:

| | 1L2Y | 3HTB |
| --- | ---: | ---: |
| SHAKE only, no COM removal | 1.766e-5 nm | 5.3e-6 nm |
| SETTLE + SHAKE, no COM removal | 1.753e-5 nm | 5.2e-6 nm |
| **SETTLE + SHAKE + COM removal** | **5.7e-9 nm** | **5.7e-9 nm** |

5.7e-9 nm is f64 round-off through the constraint solve. Both omissions were
real, and *either one alone leaves a residual that looks like the other's*.

🔴 **The residual got attributed to the wrong cause twice.** First to the
integrator's discretisation, then to SETTLE alone. What finally settled it was
asking whether OpenMM's correction could even be *produced* by the solver's
ansatz: SHAKE can only move atoms along mass-weighted reference bond
directions, and projecting OpenMM's correction onto that span left 1.892e-5 nm
outside it. A span that cannot represent a uniform translation, and a leftover
the size of `|v_com| · dt`, named the centre-of-mass remover without guessing.

The lesson is cheap to state and was not cheap to learn: when a discrepancy
will not move as you tighten the thing you believe causes it, stop tightening
and ask what *kind* of difference it is.

### 🔴 A whole step can hide a broken velocity constraint

The device's first RATTLE was not approximate, it was absent: the iterative
kernel's cluster tables hold only the *iterative* clusters, so 688 of 1L2Y's
784 -- every rigid water -- got no velocity projection at all. Both integrators
still reproduced OpenMM's trajectory step for step, to 1.0e-7 nm and 5.1e-5
nm/ps.

They did because LFMiddle repairs it. The velocities go into the drift wrong,
the positions land off the constraint surface, SHAKE pulls them back, and
`v += (x' - x_unconstrained)/dt` folds that correction into the velocity --
reconstructing most of what RATTLE should have removed. The error only survives
where the thermostat separates the two half-drifts, so it showed up at friction
1 and nowhere else: 2.36 nm/ps after the velocity solve, 2.35e-3 nm/ps by the
end of the step, with the positions still agreeing to 9.1e-8 nm.

Two things follow, and the second is the general one.

A thermostatted step has to be tested, not just a friction-free one. That needs
a generator the host can reproduce, which is why `src/reference/random.ts` is
counter-based (see also deviation 15).

And an end-to-end comparison measures the *composition* of the stages, which
can be far better conditioned than any stage in it. The number that found this
came from comparing each dispatch against the reference's corresponding
intermediate -- `stageThermostat` in src/gpu/validate-integrator.ts, which is
now a permanent part of the report for exactly this reason. With SETTLE's exact
velocity solve on the device the same step reads 4.6e-5 nm/ps, the same floor
as every other number here.

### The clusters are small, and that is measured### 🔴 A whole step can hide a broken velocity constraint

The device's first RATTLE was not approximate, it was absent: the iterative
kernel's cluster tables hold only the *iterative* clusters, so 688 of 1L2Y's
784 -- every rigid water -- got no velocity projection at all. Both integrators
still reproduced OpenMM's trajectory step for step, to 1.0e-7 nm and 5.1e-5
nm/ps.

They did because LFMiddle repairs it. The velocities go into the drift wrong,
the positions land off the constraint surface, SHAKE pulls them back, and
`v += (x' - x_unconstrained)/dt` folds that correction into the velocity --
reconstructing most of what RATTLE should have removed. The error only survives
where the thermostat separates the two half-drifts, so it showed up at friction
1 and nowhere else: 2.36 nm/ps after the velocity solve, 2.35e-3 nm/ps by the
end of the step, with the positions still agreeing to 9.1e-8 nm.

Two things follow, and the second is the general one.

A thermostatted step has to be tested, not just a friction-free one. That needs
a generator the host can reproduce, which is why `src/reference/random.ts` is
counter-based (see also deviation 15).

And an end-to-end comparison measures the *composition* of the stages, which
can be far better conditioned than any stage in it. The number that found this
came from comparing each dispatch against the reference's corresponding
intermediate -- `stageThermostat` in src/gpu/validate-integrator.ts, which is
now a permanent part of the report for exactly this reason. With SETTLE's exact
velocity solve on the device the same step reads 4.6e-5 nm/ps, the same floor
as every other number here.

### The clusters are small, and that is measured

Connected components of atoms joined by a constraint, with
`constraints=HBonds` and `rigidWater=True`:

| cluster shape | 1L2Y | 3HTB |
| --- | ---: | ---: |
| 3 atoms, 3 constraints (rigid water) | 688 | 8021 |
| 2 atoms, 1 constraint | 50 | 467 |
| 3 atoms, 2 constraints | 38 | 261 |
| 4 atoms, 3 constraints | 8 | 112 |

**The largest is 4 atoms and 3 constraints**, in both. So a GPU kernel can hold
a whole cluster in registers and iterate it locally, with no atomics and no
cross-thread coupling. A system built with `constraints=AllBonds` would not have
that property, which is why the engine checks the cluster size and refuses
rather than silently truncating.
