# Minimisation

What `LocalEnergyMinimizer.minimize` actually does, what it does not do, and
where this engine stands against it. Transcribed from
`platforms/reference/src/SimTKReference/ReferenceMinimize.cpp`, which was fetched
and read rather than recalled.

## The objective is not the potential energy

Constraints are not imposed during the search. They are turned into springs and
added to the energy:

    E(x) = U(x) + sum over constraints of (k/2) (r - d)^2

with `k = 100 / max(1e-4, constraintTolerance)`. For the notebook's 1e-6 the
**1e-4 floor wins**, so the springs are 1e6 kJ/mol/nm² and not 1e8. An outer
loop multiplies `k` by ten until every constrained distance is inside
`max(1e-4, tolerance)`, and a final *exact* projection then tightens them to
whatever the caller asked for.

That split matters: the springs never deliver the requested tolerance, they only
get the configuration close enough for the projection to finish the job.
Measured on 1L2Y: the springs reach 4.0e-5 and the projection takes it to
3.1e-8.

Replicating this objective is what makes our minimum OpenMM's minimum. The line
search only decides how fast it gets there, which is why
`src/reference/lbfgs.ts` does not try to be bit-identical to libLBFGS.

## 🔴 OpenMM does not reach its own tolerance

The documented default is 10 kJ/mol/nm on the root-mean-square force. On 1L2Y
the converged run stops at **1080**.

The reason is in the outer loop: it **ignores libLBFGS's return code** and breaks
as soon as the constraints are satisfied. A line search that stalls therefore
ends the minimisation silently, and with springs at 1e6 against soft modes
around 100 the objective is stiff enough that it does stall. The energy still
falls by 16,221 kJ/mol, which is the part that matters.

A test written to expect the documented tolerance would have been written to
fail. This is also why the engine reports `status` from the search rather than a
boolean: `line-search-failed` is the normal outcome, not an error.

### And the springs are not what keeps that gradient large

The first explanation offered for a 1000 kJ/mol/nm residual was that at a
constrained minimum the potential gradient balances the constraint forces, so it
need not be small. That is true in general and **false here**: at the end the
constraint error is 1e-5 relative, so `dr` is 1e-6 nm and a 1e6 spring
contributes about 1 kJ/mol/nm. Measured, the objective gradient and the raw
potential gradient agree to four figures — 1047.05 against 1047.09.

The minimisation simply stops short of a stationary point.
`tools/capture_minimisation.py` records both quantities so that neither story
has to be taken on trust.

## 🔴 Giving up means returning the starting coordinates

If stiffening the springs stops reducing the worst constraint error, OpenMM
resets the context to the initial positions **before** testing whether to give
up — so abandoning the search discards it. A caller whose coordinates come back
unchanged has not hit a no-op. `MinimiseResult.gaveUp` says so.

## Where this engine stands

L-BFGS runs on the host and the forces run on the device, which is OpenMM's own
architecture: its minimiser sets a context's positions and asks for forces and
energy on every evaluation. The search is a few vector operations over 3N
numbers against a force evaluation of forty-odd dispatches.

Energies are compared at **matched iteration counts**, not at convergence. A
minimum is not unique, so from a high-energy start two searches with the same
objective and different floating-point order drift into different basins; only a
shared path compares tightly, and what it pins is the objective.

Relative energy difference from OpenMM, on the device:

| iterations | 1 | 2 | 5 | 20 |
| --- | --- | --- | --- | --- |
| 1L2Y, vs Reference | 9.1e-5 | 3.5e-5 | 4.8e-5 | 1.9e-3 |
| 3HTB, vs CPU | 8.5e-6 | 2.7e-5 | 4.3e-4 | 5.0e-4 |

The double-precision reference lands 9.2e-5, 3.6e-5 and 4.8e-5 on 1L2Y — the
same as the device to two figures, which is the clearest evidence available that
the growth at twenty iterations is divergence between searches rather than error
in either. At twenty iterations the reference is 6.5e-3 from OpenMM and the
device is 1.9e-3: **the device is closer**, which only makes sense if both are
simply elsewhere.

3HTB diverges about nine times faster than 1L2Y, which tracks its direct-space
force relative RMS being 1.13e-5 against 7.65e-7 (the open item in
[NONBONDED.md](NONBONDED.md)). A line search turns a force difference into a
different backtracking decision and then a different path.

A long run, 200 iterations, from each fixture's own freshly built state:

| | start | after | potential \|f\| rms | constraint error |
| --- | --- | --- | --- | --- |
| 1L2Y | -21,532 | **-33,871** | 955 | 4.3e-8 |
| 3HTB | -204,407 | **-410,648** | 1,024 | 6.4e-8 |

## What the position format costs

Coordinates are fixed point with a resolution of 2^-32 nm, and a backtracking
line search halves its trial step until the energy improves. Below that
resolution the device sees no movement at all, so the search must stall earlier
than a double-precision one would.

Measured, it does not bite before the line search stalls for its own reasons:
the device reports `line-search-failed` on 1L2Y at twenty and two hundred
iterations, which is the same outcome OpenMM reaches, and the energies above are
what it reaches. This is recorded because it is a real limit of the format that
happens not to be the binding one, which is worth knowing before someone
tightens a tolerance and wonders why nothing moves.
