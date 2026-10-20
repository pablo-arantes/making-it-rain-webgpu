# Where this left off

`AGENTS.md` has the invariants and `CLAUDE.md` the durable findings. This file
has only what is **unfinished**.

## State

**Phase 0 is done. Phase 1 is underway**, through the first of its six steps.

`npm run check` passes: typecheck, 45 unit and reference tests, web build.
`npm run test:browser` passes all five device checks plus the bonded-force
comparison on both fixtures. `npm run bench` and `npm run report:adapter` work
and their output is committed under `docs/measurements/`.

Done in Phase 1 so far:

- **`System` and `State` parsers** (`src/engine/system.ts`, `state.ts`, over a
  purpose-built scanner in `xml.ts`): 7.5 MB of `system.xml` in 39 ms, every
  force type this workflow produces, with an explicit named error for anything
  else.
- **Bonded forces**, reference and WGSL: energies match OpenMM to between
  1.7e-8 and 1.6e-6 relative on both fixtures, forces to between 5.6e-7 and
  7.1e-6 relative RMS.
- **Direct-space Ewald and Lennard-Jones**, reference and WGSL, including the
  exclusion corrections, the exceptions, and the analytic dispersion
  correction. On the GPU: 1.05e-7 energy agreement with OpenMM on 1L2Y and
  **3.93e-7 on 3HTB's 26,749 atoms**, where +1.67e6 kJ/mol of exclusion
  corrections nearly cancel against a +1.56e6 total.
- **Reciprocal-space PME**, reference and WGSL: order-5 B-spline spreading, a
  mixed-radix Stockham FFT over radices 2/3/5/7, the convolution and
  interpolated forces. On the GPU, 1.23e-8 energy agreement with OpenMM on
  1L2Y's 25³ mesh and 1.36e-7 on 3HTB's 54³. The transform has its own
  differential test against the f64 reference, which is itself checked against a
  direct O(n²) transform.
- **Every force term of the notebook's `System` now runs on the GPU** and
  agrees with OpenMM.
- **erf and erfc**, f64 for the reference and a rational approximation for the
  kernel, the latter measured at < 1.2e-7 relative over the whole Ewald range.
- **The nonbonded decomposition, measured and written down** in
  `docs/NONBONDED.md`. Three terms are not what a first reading suggests, and
  each is larger than the total energy or would have silently biased it.
- **Constraints: SHAKE and RATTLE, reference only**, matching OpenMM's
  algorithm including its simultaneous (Jacobi) delta application. Agrees with
  `Context.applyConstraints` to 1.8e-7 nm on 1L2Y and 6.2e-7 on 3HTB, which is
  the width of the tolerance band.
- **Two deterministic trajectory oracles**, velocity Verlet and LFMiddle at
  zero friction, with `tools/capture_constraints.py`, carrying positions,
  velocities and forces per frame.
- **A buffer pool** (`src/gpu/allocator.ts`), the single owner `AGENTS.md`
  requires, with named buffers that persist and leases that get reused. Tested
  against a stub device, so it is covered in the lane that runs everywhere.
- **The constraint kernels**: SETTLE analytically and SHAKE/RATTLE iteratively,
  one thread per cluster, no atomics. Within 8.7e-8 nm of the reference on 1L2Y
  and 9.6e-8 on 3HTB, leaving the constraints satisfied to 1.0e-6 -- exactly the
  tolerance they were asked for. Positions and velocities go through separate
  entry points, because on the device they are separate formats.
- **Both integrator kernels**, against OpenMM's captured trajectories, driven by
  the forces OpenMM recorded at each frame so nothing from the force kernels
  leaks in. Verlet 9.6e-8 nm and LFMiddle 1.0e-7 nm a step on 1L2Y, 1.0e-7 and
  1.1e-7 on 3HTB, velocities 5.3e-5 nm/ps -- which is the position agreement
  divided by the timestep, so the whole budget is the constraint solve.
- **A thermostat that is tested with the noise switched on**, which needed a
  counter-based generator: Threefry-2x32-20, bit-identical to Random123's
  published known answers on the host and to the host on the device. At friction
  1 a step lands 4.6e-5 nm/ps from the reference.
- **Centre-of-mass removal and the kinetic energy**, reduced in fixed point so
  the total does not depend on the order the workgroups landed in. 4.8e-10 nm/ps
  and 2.1e-8 relative.
- **The cell list, built on the device** as a counting sort and then sorted
  within each cell. It puts every atom in the cell the host builder puts it in
  (0 differences on both fixtures) and gives bit-identical forces -- which it has
  to, and which is not free: see below.
- **Both integrators, SETTLE, SHAKE/RATTLE and centre-of-mass removal**, all in
  the reference and transcribed from OpenMM. A step reproduces OpenMM's to
  **5.7e-9 nm** on both fixtures -- f64 round-off through the constraint solve --
  tested step by step against both oracles and driven by **OpenMM's own recorded
  forces**, so a force-field error cannot leak into an integrator test.
- **Both kinetic-energy conventions** (`src/reference/kinetic.ts`), reproducing
  what OpenMM reports to 1e-12 unshifted and 1e-5 shifted. This is a parity
  requirement for the notebook's log files, not a detail.
- **The dynamics algorithms written down** in `docs/DYNAMICS.md`, with five
  things a first reading gets wrong.

**Phase 1 is complete.** A whole NPT step runs on the GPU -- forces,
constraints, both integrators, the thermostat, centre-of-mass removal,
restraints, the barostat, the minimiser -- and every piece is differentially
tested against OpenMM. What is left is Phase 2: a user interface over this.

The barostat is host-driven, which is where OpenMM's API drives it too: read the
energy, rescale the box and the molecules, read it again, accept or reject. At
frequency 25 the position round trip is about 1 ms against 125 ms of stepping,
so it is not where the time goes -- the two force evaluations are.

| | molecules | setBox vs construction | constraint change | accepted | volume |
| --- | ---: | ---: | ---: | ---: | --- |
| 1L2Y | 694 | **0** | **0** | 3/4 | 29.24 → 28.59 nm³ |
| 3HTB | 8073 | **0** | **0** | 1/4 | 285.29 → 283.23 nm³ |

The box is now dynamic, which is the part that needed care: changing it rewrites
the pair and exception kernels' boxes in f32 *and* fixed point, three
reciprocal-space blocks whose `prefactor` carries 1/volume, the cell geometry,
each restraint's wrap, and the two host-side energy terms that depend on volume.
Missing one of those is not a crash; it is an energy slightly wrong in a way
that looks like ordinary single-precision disagreement. So `setBox` is checked
against a loop *constructed* at that box -- the two reach every uniform by
different routes -- and asserted **exactly equal**, which needs no tolerance and
costs nothing. It used to compare against the f64 reference at the new box,
which was both weaker and minutes slower per fixture.

Two things deliberately do not change with the box. The **PME mesh** stays as
the starting box chose it, which is what OpenMM does. And the **cell divisions**
stay, because the number of cells sizes the buffer the pair kernel indexes;
the extents move instead, and `BoxTooSmallForCells` is thrown rather than
quietly returning a list that misses pairs. The headroom is 8.7% for 3HTB.

Restraints are done: `k*periodicdistance(x,y,z,x0,y0,z0)^2`, implemented
directly rather than through an expression parser. Against OpenMM's own force
group at three configurations, the third of which displaces every atom by a
whole box to exercise the minimum image -- 154 restrained atoms on 1L2Y and
1311 on 3HTB, matching the heavy-atom mask count in deviation 1. Worst force
error 1.7e-4 kJ/mol/nm against forces of 2152, and the energy agrees to 1.9e-9
relative at 9.8 million kJ/mol.

After those, Phase 1 is complete and Phase 2 -- upload a system and run it -- is
a user interface over what exists.

The minimiser is done. L-BFGS on the host with the forces on the device, which
is OpenMM's own architecture, and OpenMM's objective exactly: constraint springs
at `100/max(1e-4, tol)` growing tenfold, then an exact projection. Against
OpenMM at matched iteration counts (docs/MINIMISER.md):

| iterations | 1 | 2 | 5 |
| --- | --- | --- | --- |
| 1L2Y | 9.1e-5 | 3.5e-5 | 4.8e-5 |
| 3HTB | 8.5e-6 | 2.7e-5 | 4.3e-4 |

Two hundred iterations takes 1L2Y from -21,532 to **-33,871** kJ/mol and 3HTB
from -204,407 to **-410,648**, with the constraints holding to 4.3e-8 and
6.4e-8 -- from states that violate them by 25%.

The step loop runs. On an M3 Pro through Chrome 155, from the captured
equilibrated frame, velocity Verlet at 2 fs with the cell list rebuilt every
step:

| | passes/step | two runs differ by | one step vs reference | drift | ns/day | peak pool |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1L2Y, 2,373 atoms | 28 | **0** | 8.08e-8 nm | 0.282 | 39.1 | 1.0 MB |
| 3HTB, 26,749 atoms | 40 | **0** | 9.30e-8 nm | 0.540 | 10.4 | 9.9 MB |

Drift is kJ/mol per atom per ns from a least-squares slope over ten reports, and
the reports are inside the timed run, so ns/day is a reporting cadence rather
than a bare loop. The number to read the drift against is OpenMM's own, fitted
the same way from the same configuration and integrator on its CPU platform:
**0.374** and **0.385** (`tools/measure_drift.py`). 1L2Y is better than that and
3HTB is within 40%.

🔴 An endpoint difference is not a drift. This used to subtract the first report
from the last, and when the forces got *more* accurate that number went from
0.043 to 0.426 -- over a picosecond the total energy fluctuates by about as much
as it trends, so two points cannot tell one from the other. The Python tool had
fitted a slope from the start and this had not been brought into line with it. After those, Phase 1's
gates: NVE drift at 2 fs, and a measured ns/day.

A whole step now runs on the GPU -- forces, constraints, both integrators, the
thermostat, centre-of-mass removal and the reported kinetic energy -- and
reproduces OpenMM's own trajectory to 1.1e-7 nm a step. What remains is to stop
taking it apart between dispatches: `src/gpu/integrate.ts` still builds a
module, a pipeline and a bind group per phase and drains the queue after each,
which is six sync points a step. That is the loop's job, and the allocator and
the declared bind group layouts are already in place for it.

## Open, in the order they are worth doing

### The FFT is correct but not yet tuned
Radices 2, 3, 5 and 7, with a generic R² inner product per butterfly rather
than specialised butterflies — at most 49 terms, since the radix is at most 7.
Specialised radix-2/3/4/5 butterflies and a radix-4 or radix-8 path for
power-of-two sizes are the obvious next step, and the differential test in
`src/gpu/checks.ts` is what they would be held to. **Measure before writing
them**: the per-step cost of PME against everything else is not known yet,
because there is no step loop to measure.

### 🔴 Three mistakes the barostat made, all of the same kind
Each was a quantity measured against the wrong baseline, and each produced a
plausible-looking number rather than an error.

**Running from `state.xml`.** It is un-minimised -- 5.6e5 kJ/mol/nm on 3HTB --
so a hundred steps of NVE from it diverges. The symptom was "2.15e9 is outside
the fixed-point range" for a *position*: a blown-up simulation, reported as a
format limit. Dynamics tests start from the captured equilibrated frame. This is
the second time this caught me, after the restraint drift check below.

**Measuring the constraint error instead of its change.** The fixture violates
its own constraints by 25% before anything runs, so an absolute bound measures
the fixture. What rigid-molecule scaling claims is that internal geometry is
untouched, so the violation must come out exactly as it went in -- which it
does, to 0.

**Advancing the barostat counter by one per batch.** `barostatActsNow` is
OpenMM's per-step form, and a step loop encodes many steps into one command
buffer, so the host reaches the barostat once per batch. It gave zero attempts
in a hundred steps at frequency 25. `barostatActsAfterSteps` advances by the
batch and carries any overshoot.

And one diagnostic that lied: `fixedWords` named "kJ/mol/nm" in its range error
while also converting positions, forces, energies and box lengths. It now names
none of them.

### A rigid translation isolates one force inside a total
Checking that the restraint pass is actually *encoded into a step* -- rather
than prepared and forgotten, which every single-point comparison would still
pass -- needs a configuration where the restraint is a measurable share of a
well-behaved total. The wrapped fixture is not it: 3 nm displacements overlap
atoms and the total comes out at 3.9e14 kJ/mol, where f32 and f64 have nothing
in common.

Shifting every atom by the same vector works instead. Every distance, angle and
minimum-image separation is untouched, so the force field is unchanged, while
every restrained atom moves exactly that far from its reference point -- and the
restraint's contribution is then known in closed form. At 0.1 nm it is 1078
kJ/mol out of -20,454 on 1L2Y, and the loop's assembled total matches the
reference's to 5.9e-7. Deviation 7 used as an instrument.

An earlier attempt measured drift of restrained against free atoms over 200
steps and got a ratio of 1.02 -- not because anything was wrong but because the
run starts un-minimised, where the driving forces are ten times the restraint's.
A statistical claim needs an equilibrated system and many steps; an energy
comparison needs neither.

### 🔴 OpenMM's minimiser does not reach its own tolerance
Its documented default is 10 kJ/mol/nm on the RMS force; on 1L2Y it stops at
1080, because the outer loop ignores libLBFGS's return code and breaks as soon
as the constraints are satisfied. A stalled line search therefore ends the
minimisation silently. The energy still falls by 16,221 kJ/mol. Anyone writing
a convergence assertion against the documented number will write a failing test.
docs/MINIMISER.md has that and two other behaviours worth knowing -- including
that OpenMM *discards the minimisation and returns the starting coordinates* if
stiffening the springs stops helping.

### ✅ 3HTB's direct-space force error: closed, and how the estimate misled
It was 1.13e-5 relative RMS against 1L2Y's 7.65e-7, for the same kernel. The
cause was the order of the minimum-image wrap: the pair difference was converted
to f32 while it could still be box-sized, rounding it at 4.8e-7 nm, and the wrap
that followed could not recover what the conversion dropped. Wrapping in fixed
point first took 3HTB to **5.37e-7** -- a factor of 21, and now better than
1L2Y. The drift gate followed, from 0.96 to 0.540 against OpenMM's 0.385.

Two wrong turns are recorded in docs/NONBONDED.md because both were settled by
measurement rather than argument. Cancellation in the per-atom f32 sum was
offered first and **disproved** by making that sum exact and watching the number
not move (it did improve 1L2Y's drift, by removing the same error from the
reported *energy*, which is a different thing). And the bound derived from the
pair-vector measurement -- "about 3 of the 13" -- was wrong by an order of
magnitude, because it assumed a vector error passes through to a force at unit
gain; the gain is `dF/dr`, about 1e5 kJ/mol/nm^2 at short range.

It costs about 25% of the throughput, which is recorded rather than hidden.

### 🟡 One unexplained run, and the instrument that would name it next time
A single `test:browser` run had 3HTB's LFMiddle step 0.0167 nm from OpenMM --
while still matching our own f64 reference to 1.07e-7, and with every other
number unchanged. That combination is the signature of a **shared bad input**:
the reference is deterministic given its inputs, so if it and the device agree
and both disagree with OpenMM, what they were handed differed. Five subsequent
runs were clean and the fixtures' checksums were byte-identical every time, so
whatever happened was in the transport and not on disk.

It is not reproduced and not explained, so it is not called fixed. What is in
place is the instrument: the page checksums each trajectory array as it fetches
it and reports it, and `test/browser/bonded.spec.ts` reads the same files off
disk and compares. If it recurs, the failure says which file arrived wrong
instead of looking like a physics regression -- which is the part that cost time
here, because 0.0167 nm is exactly the size of a plausible bug.

### An exact accumulator does not make a kernel order-independent
The forces accumulate in fixed point, so the sum across threads is exact. But
the pair kernel sums one atom's neighbours in an f32 register before the single
exact add at the end, and that part reassociates. Measured at 1.89e-3 kJ/mol/nm
between two orderings of the same partition -- 2e-6 of the force RMS, invisible
in a force comparison and enough to make two runs differ. The cell list is
sorted for that reason, `AGENTS.md`'s determinism bullet is amended, and
docs/NONBONDED.md has the measurement. The same section covers why the pair
kernel now reads its home cell from the list instead of recomputing it in f32.

### A step can hide a broken stage
The device's first RATTLE did nothing at all to the rigid waters -- the
iterative kernel's tables hold only the iterative clusters -- and both
integrators still reproduced OpenMM step for step. LFMiddle repairs it: the
position solve and the velocity correction that follows reconstruct most of what
the velocity solve should have removed. It showed only at friction 1, and only
once each dispatch was compared against the reference's corresponding
intermediate. That staged comparison is now part of the report
(`stageThermostat`), and the lesson is in docs/DYNAMICS.md: an end-to-end number
measures the composition of the stages and can be far better conditioned than
any one of them.

### A reserved-word scan now guards the kernels
WGSL reserves a long list of words it does not yet use. Tint enforces only some
of them -- `reference` compiles, `from` and `target` do not -- so this is
precisely the class of bug that passes on one backend and fails on the next.
`test/wgsl-reserved.test.ts` scans every kernel, needs no GPU, and went in after
two of these each cost a round trip.

### The kinetic-energy convention is per integrator
`VerletIntegrator` reports kinetic energy from velocities shifted by +dt/2 and
then RATTLE'd; the Langevin integrators report it unshifted. The shifted one is
the conserved quantity for a leapfrog integrator -- 3.1e-3 against 2.5e-2
kJ/mol an atom over 40 steps of 1L2Y -- so the NVE gate uses it. But the
notebook logs with `LangevinIntegrator`, so **the reporters must use the
unshifted convention** or every log line disagrees. Both are implemented and
tested; the thing still to do is wire them to the right integrator in the run
loop, and record which was used in `run.json`.

### The acceptance thresholds are provisional
`AGENTS.md` quotes OpenCL **single** precision on an Apple M3 Pro, because this
machine's OpenMM has no mixed-precision GPU platform. Phase 1's gates need the
real thing: `tools/calibrate_thresholds.py` on the **Colab T4**, for 1L2Y, 3HTB
and DHFR. Pablo has agreed to run it. Until then every Phase 1 threshold is a
placeholder with a plausible value, which is the worst kind.

### Whether the nonbonded kernel gathers or scatters
Settled for determinism, open for speed. An atomic scatter is bit-reproducible
because the accumulation is integer, so gather is now a pure performance
question — and on Apple the atomics are cheap (374 M adds/s into one contended
slot, *faster* than 65536 uncontended ones, so scattered memory costs more than
contention). Measure both on NVIDIA and on an integrated adapter before
committing the nonbonded kernel either way.

### The oracle lane has no oracle
`npm run test:oracle` reports `vendor/openmm-wasm/openmm.wasm` missing and skips.
Building it is Phase 1 work, from the recipe in `docs/PRIOR_ART.md` §1 — OpenMM
8.2, Emscripten, Reference platform only, no pthreads, plus the serial-CCMA
patch without which a constrained `Context` deadlocks in the browser. About
906 KB when built. It is the acceptance reference for every force kernel *and*
the "verify this system" button, so it is worth doing early rather than late.

### Dawn does not load on this machine
`npm run test:gpu` cannot run on macOS 14 — every published binary wants
macOS 15+. The browser lane covers it, and CI runs Dawn on Linux, so this is
recorded rather than blocking. It will resolve itself on a macOS upgrade.

### Smaller
* `src/gpu/compute.ts` allocates buffers directly. `AGENTS.md` requires every
  allocation to go through one place so peak memory stays measurable; there is
  nothing to pool yet, and Phase 1 must replace it before the force kernels
  land.
* The three CI jobs other than `check` are `continue-on-error: true`, because
  none has been run on a hosted runner yet. Tighten each as it proves itself;
  leaving them permanently soft would make them decorative.

## Next

Phase 1, in the order the plan gives: `System`/`State` parser, then the
TypeScript double-precision reference, then WGSL bonded → direct space with
exceptions → PME reciprocal → constraints → integrator → barostat, each against
the committed fixtures.
