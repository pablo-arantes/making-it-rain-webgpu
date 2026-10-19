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
784 in 1L2Y, 8021 of 8861 in 3HTB.

SETTLE is analytic: it computes the rigid-body placement that satisfies all
three distances exactly. SHAKE is iterative and moves atoms along the reference
bond directions. Both end up on the constraint manifold, and **they end up at
different points on it.** Measured over one 2 fs step of 1L2Y, our SHAKE lands
1.75e-5 nm from OpenMM's positions, and the worst-offending atoms are all
waters.

The diagnosis was that tightening our own tolerance does nothing:

| our tolerance | iterations | distance from OpenMM's next frame |
| --- | ---: | ---: |
| 1e-6 | 40 | 1.766e-5 nm |
| 1e-8 | 58 | 1.753e-5 nm |
| 1e-10 | 76 | 1.753e-5 nm |
| 1e-12 | 94 | **1.753e-5 nm** |

A solver stopping too early would converge towards agreement. This does not, so
the two are solving different problems. OpenMM's own frame satisfies the
constraints to 3.1e-8 relative — far inside its 1e-6 tolerance — which is what
an analytic solution looks like.

So `docs/PLAN.md`'s "SETTLE for water, CCMA-style for bonds to H" is a
**parity requirement**, not the performance note it reads as. Until SETTLE is
implemented, the step-by-step trajectory gate is held at 5e-5 nm instead of the
1e-7 the tolerance would allow.

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
