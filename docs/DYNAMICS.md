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

### 🔴 LFMiddle at zero friction does not conserve energy

With `friction = 0`, `vscale = 1` and `noisescale = 0`, so the thermostat is
the identity and the step becomes `v += dt·F/m` then `x += dt·v`. That is
**symplectic Euler, not velocity Verlet**. It conserves a *modified*
Hamiltonian, so the naive potential-plus-kinetic energy oscillates at O(dt)
amplitude instead of drifting.

Measured on 1L2Y, equilibrated, total energy change over 8 steps per atom:

| dt (fs) | Langevin, friction 0 | velocity Verlet |
| ---: | ---: | ---: |
| 2.00 | +3.47e-2 | −2.7e-4 |
| 1.00 | −3.09e-2 | −6.4e-4 |
| 0.50 | −3.11e-2 | −3.7e-5 |
| 0.25 | −1.16e-2 | −1.2e-4 |

Over 40 steps at 2 fs the Langevin series reads +0.000, +0.017, +0.014, +0.000,
+0.022, +0.015, +0.025, +0.009, −0.004 — bouncing, with no trend, and not
shrinking cleanly with the timestep. Verlet on the same configuration stays
inside 3e-3 and does fall with dt.

This is by design: LFMiddle exists for thermostatted *sampling*, where it
samples configurations better than velocity Verlet at a given timestep. It is
not an NVE integrator.

**Consequence for Phase 1's gates.** The NVE energy-drift gate must use
velocity Verlet, which is what `tools/calibrate_thresholds.py` already does —
and why the threshold it measured is 0.17 kJ/mol/ns/atom rather than something
fifty times larger. LFMiddle at zero friction is still the right oracle for
**step-by-step determinism**, because it *is* deterministic; it just is not
conservative. Both trajectories are therefore captured, and the engine
implements both integrators.

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
