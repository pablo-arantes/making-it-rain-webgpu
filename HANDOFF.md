# Where this left off

`AGENTS.md` has the invariants and `CLAUDE.md` the durable findings. This file
has only what is **unfinished**.

## State

**Phases 0 to 4 are done. Phase 5 is underway.** A PDB ID or an upload becomes a
runnable `System` in the browser, protein or protein–ligand, runs on the GPU
against OpenMM-calibrated thresholds, and comes back as a DCD with the
notebook's figures.

`npm run check` passes: typecheck, **672 unit and reference tests** over 48
files (9 more are skipped: the Dawn and oracle lanes), and the web build. `npm run test:browser` passes **13 tests** in Chrome —
the device and precision checks, bonded forces on both fixtures, the three
preparation paths, the step profile, and the run page end to end. The 1 ns gate
is the 14th and is skipped unless `MIR_LONG_RUN=1`; it passes in **7.6 minutes**.
`npm run bench` and `npm run report:adapter` work and their output is committed
under `docs/measurements/`.

🔴 **This file was wrong about that for five commits**, claiming Phase 1 was on
its first step and that there were 45 tests, while Phase 4 landed and Phase 5
started. A handoff that overstates what is left is survivable; one that
*understates* it sends the next person to rebuild what already works. Check the
section below against `git log` before trusting it — and the test count has
since gone 45, 566, 628, 662, 672, so treat that number as the first thing
to re-measure rather than quote.

Done in Phase 1, which is where the engine's parity record lives:

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

**Phase 1 is complete**, and Phase 2 has its engine side working: **a whole run
goes end to end in a browser.** Minimise with the restraints on, draw
velocities, equilibrate, then produce by strides -- writing a DCD, a log and a
checkpoint for each, plus `run.json`:

| | ensemble | stages | log steps | temperature | ns/day | artefacts |
| --- | --- | --- | --- | --- | ---: | ---: |
| 1L2Y | NVT | equil + 2 strides | 500–2000, 2500–4000, 4500–6000 | 291–314 K | 38 | 10 |
| 3HTB | NPT | equil + 2 strides | same | 295–316 K | 9.3 | 10 |

The step numbers continuing across strides is the thing to notice: reports land
on multiples of the interval measured on the *absolute* counter, which is what
keeps a log lining up with its trajectory. A second pass over the same outputs
skips every stage.

🔴 **1L2Y cannot run NPT at a 1 nm cutoff**, and that is not a bug in the
barostat. The run page now says so before starting rather than failing during. Its box is 3.0806 nm -- 3.08 cutoffs -- so cells start at 1.0269 nm
and **2.6% of compression** takes them below the cutoff, after which the 27-cell
scan stops finding every pair. Fewer cells do not help: below three per axis the
scan visits the same cell twice. This is finding 4 in docs/NOTEBOOK_PARITY.md
turning into a hard limit -- that fixture's solute already sees its own image at
0.958 nm. The validator measures the headroom and picks the ensemble rather than
assuming, and an **all-pairs fallback for tight boxes is the next thing the
engine needs**.

**Phase 2 is built.** `web/run.html` takes an OpenMM `System` and `State`,
checks them, runs the protocol and hands back the artefacts; nothing leaves the
tab. Driven end to end under Playwright through the real file inputs:

| | measured |
| --- | --- |
| verify, 2,373 particles | 5.0e-7 energy, 1.2e-6 force relative, 0.7 s |
| a three-stage run | 13 artefacts, 1.9 MB, 6 reports |
| OPFS vs memory | byte-identical (1.3 MB and 15 MB) |
| device lost and resumed | skipped the two finished stages, redid one |
| DCD readers | mdtraj, pytraj, MDAnalysis 2.9.0 all within f32 round-off |

**"Verify this system"** compares the kernels against `src/reference/` rather
than against OpenMM, because a stranger's upload has no captured energies. That
is weaker and it catches what a fixture cannot: parameters outside the tested
range, a box that defeats the cell list, molecules split across the boundary. It
also answers, *before* a run rather than forty minutes into one, whether the box
can take a barostat.

What remains for Phase 2's release: deploy to GitHub Pages marked experimental,
and the plan's remaining gate items beyond the 1 ns run — a **prmtop/inpcrd**
reader, so a user with Amber files does not have to go through OpenMM first.

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

### The FFT is correct but not yet tuned — and now there is a number
Radices 2, 3, 5 and 7, with a generic R² inner product per butterfly rather
than specialised butterflies — at most 49 terms, since the radix is at most 7.
Specialised radix-2/3/4/5 butterflies and a radix-4 or radix-8 path for
power-of-two sizes are the obvious next step, and the differential test in
`src/gpu/checks.ts` is what they would be held to.

This entry used to end "**measure before writing them**: the per-step cost of
PME against everything else is not known yet, because there is no step loop to
measure." It is now measured, from a *minimised* state, on an M3 Pro
(`docs/measurements/profile-step-m3pro.json`):

| | 1L2Y | 3HTB |
| --- | --- | --- |
| pair force | 35% | 39% |
| neighbour-list build | 13% | 30% |
| **PME** | **30%** | **24%** |
| everything else | 22% | 7% |

So PME is worth about a quarter of a step and no single kernel dominates any
more. Of that quarter, roughly 40% is the FFT passes and 35% is `pme spread`;
`zero/pmeGrid` and `pme load grid` together are about 4% of the whole step and
are pure bookkeeping, which is the cheapest thing on this list to attack.

🔴 Profile from a **minimised** state or the attribution is wrong, not just the
total: an unrelaxed fixture is a diverging system, and a diverging system
reallocates the step toward whatever walks neighbours. Measured from 3HTB's raw
fixture, the build looked like 87% of a step and the step looked ten times
slower than a timed run had just shown it was.

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

### ✅ An exact accumulator *does* now make the pair kernel order-independent
This entry said the opposite, and was right when written. The pair kernel then
summed one atom's neighbours in an f32 register before a single exact add, and
that part reassociated -- 1.89e-3 kJ/mol/nm between two orderings of the same
partition, 2e-6 of the force RMS, invisible in a force comparison and enough to
make two runs differ.

`accumulate_pair` now converts and adds **every pair** in fixed point, so
nothing reassociates. The evidence is `validateNeighbours` rather than the
argument: the cell-shell path combines 27 partial sums an atom, the
neighbour-list path combines 8, those are different groupings of the same terms,
and they agree on all 80,247 force components of 3HTB **bit for bit**. An
inexact accumulator could not do that.

`sort_cells` stays, for the two reasons still standing and not for the force: it
keeps the device's cell list comparable to the host builder's, which is how that
list is validated at all, and it is what gives the neighbour list built from it a
defined order. `AGENTS.md`'s determinism bullet has now stated the wrong
argument twice, so it carries the history as well as the rule. The same section
of docs/NONBONDED.md covers why the pair kernel reads its home cell from the
list instead of recomputing it in f32.

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

### ✅ Nothing binds more than WebGPU guarantees, and a test now says so
`maxStorageBuffersPerShaderStage` has a mandated floor of 8 and Apple reports
far more, so a kernel binding nine would compile here, pass every differential
check here, and fail to create its pipeline on a conforming Intel or NVIDIA
device. This machine cannot catch that class of bug at all.
`test/kernel-bindings.test.ts` reads the *composed* sources, follows each entry
point's calls, and asserts the count — one check per entry point, 62 of them,
in the lane that runs everywhere. The busiest is `direct_pairs_list` at **7 of
8**, so there is one spare and the neighbour list used two of the three that
were. It also keeps `MAX_STORAGE_BUFFERS_USED` from drifting below what a kernel
uses, which is the direction that turns a passing preflight into a failed
pipeline.

### Smaller
* `src/gpu/compute.ts` allocates buffers directly, and the entry that used to
  be here said "Phase 1 must replace it before the force kernels land". They
  landed, the pool exists, and the module now argues the other way in its own
  header: these are one-shot evaluations where nothing outlives the call, so
  there is nothing to amortise, and `AGENTS.md`'s requirement is about the
  memory a *run* holds. That argument covers `uploadBuffer` and `storageBuffer`.
  It does **not** cover `readBuffer`, which the step loop calls — once a report,
  for positions, forces, energy and the Verlet flags — and which creates and
  destroys a staging buffer each time. Transient and bounded, so not urgent;
  worth pooling if a profile ever shows it, which one has not.
* The three CI jobs other than `check` are `continue-on-error: true`, because
  none has been run on a hosted runner yet — and none can be, because the repo
  has no remote. That is the blocker, not the jobs: tighten each as it proves
  itself on a runner, and until then they are decorative rather than soft.

## Phase 2's gates

The 1 ns run passes: 43.4 minutes for 1L2Y through `web/run.html`, 100 reports,
9 artefacts, served from the production build.
`docs/measurements/gate-1ns-m3pro.json` has the numbers.

The trajectory gate is met. MDTraj and pytraj agree to 2.9e-7 nm on a real
2373-particle DCD, and pytraj's read is what covers VMD too — cpptraj goes
through VMD's own `dcdplugin`, which announces `CHARMM format DCD file` on
stderr as it does so.

🔴 **pytraj needs the right environment.** `~/miniforge3/envs/ambertools` has an
x86_64 build that will not load on arm64 (`incompatible architecture`);
`parametrizani_env` works. Both `tools/crossread_dcd.py` and
`tools/pytraj_analyses.py` want that interpreter.

## Next

Phase 3 is nearly done. A deposited PDB becomes a runnable system entirely in
the browser, and the chain is closed at both ends: every force in a prepared
`System` matches OpenMM's element for element on the parity fixtures, OpenMM
reads the file this project writes and agrees on the energy to 9.2e-12, and the
GPU then agrees with our own f64 reference to 1.07e-6 on that same prepared
system (`docs/measurements/prepared-system-verify-m3pro.json`). The analyses
agree with pytraj. `docs/PREPARATION.md` has the tables.

**Phase 3 is closed.** ff99SB, DHFR (1RX2) and 1UBQ went through the parity
gate -- eight fixtures in all -- and `prmtop`/`inpcrd` export is in
`src/prep/prmtop.ts`. One item outlived the phase and is now also done, with a
tail:

1. **✅ A non-rectangular box** (deviation 19) — **done, and a dodecahedron
   runs on the GPU.** Kept here rather than deleted, because what it cost and
   what it turned up are the useful parts.

   A 5,002-atom dodecahedral 1L2Y, through the same commands a cube goes
   through: engine against its own f64 reference at 9.8e-7 energy and 8.3e-7
   force relative RMS; the cell list agreeing host-to-device with 0 assignment
   and 0 force differences; the neighbour list bit-identical to the cell-shell
   path over all 15,006 force components, and the same trajectory over 49
   rebuilds; 657 steps/s with the list against 281 without, both paths ending at
   the same energy per atom. `docs/measurements/triclinic-gpu-dodecahedron.json`.

   The cube is unchanged throughout. The only numerical movement in six commits
   was PME's, at the f32 ulp, because the kernels now compute `r · B⁻¹` where
   they divided by an edge length — recorded in
   `engine-force-error-equilibrated-m3pro.json` rather than left behind a stale
   claim of exactness.

   🔴 **The surprise, and the thing deviation 19 got wrong for months.** A
   dodecahedron holds 70.7% of the enclosing cube's water *for the same
   padding*, which is what made it look worth having. But it cannot use the same
   padding: the 27-cell scan needs three cells an axis across the
   **perpendicular** widths, and a rhombic dodecahedron's narrowest width is
   0.707 of its edge, so it needs `d >= 3*sqrt(2)*cutoff` — 4.24 nm at a 1 nm
   cutoff against the cube's 3.0. 1L2Y at the notebook's 0.4 nm padding gives
   2x2x2 cells and is refused where the cube at the same padding is accepted. At
   1.5 nm it runs, at 5,002 atoms against the cube's 2,373. **So the saving is
   real only for a solute large enough that `3*sqrt(2)*cutoff` is not the binding
   constraint, and a small one is better off in a cube.** The refusal names the
   widths and the cell count so a user is told which case they are in.

   Two findings worth more than the feature:

   * **OpenMM's *Reference* platform is wrong for a skewed cell** — it reports
     pairs genuinely inside the cutoff as non-interacting, 72 of 689 on a
     dodecahedron and 0 of 850 on a cube, and the loss scales with skew. It is
     the oracle everywhere else in this project, so the CPU platform has to be
     used instead. Finding it meant eliminating the minimum image, the cell
     list, alpha, the dispersion correction, the exceptions and the exclusion
     correction one at a time first. `CLAUDE.md` and
     `triclinic-reference-parity.json`.
   * **OpenMM 7.7 cannot solvate a dodecahedron** — it removes clashes across a
     *rectangular* boundary and produces a 9.6e10 kJ/mol structure, silently,
     with the cube unaffected. Build triclinic fixtures with the oracle
     environment.

   Still open, and smaller than it was: `src/prep/solvate.ts` still refuses a
   non-cubic `boxShape`, so the *engine* runs one but the *preparation* path
   will not build one. `PeriodicGrid` buckets in Cartesian space and needs
   fractional, exactly as the device and host cell lists now do. Until that
   lands, a triclinic system has to come from OpenMM.

**Phase 4 is done**, and it was half the project's stated scope:
protein–ligand. A heterogen picker that asks which residue is the ligand rather
than guessing, CCD bond orders, ligand parameters by ffxml upload, and the
ligand analyses, all through the parity gate at 26,749 atoms on 3HTB+JZ4. One
piece was cut rather than finished: the ProLIF interaction network, on the
grounds that the notebook's version is not what this tool is for. Typing a
ligand *in the browser* -- SMIRNOFF through RDKit in WASM, NAGL in ONNX -- was
not built either; parameters are uploaded instead, which is deviation 23.

**Phase 5 has started.** `deviceFitness` answers "will *this* system run on
*this* GPU" before any GPU work, naming the limit that binds — storage binding,
buffer size, dispatch width, storage buffers per stage, or invocations per
workgroup. `DEVICE_PROFILES` holds WebGPU's mandated floor plus integrated and
discrete shapes, so the arithmetic is tested against hardware this machine does
not have; `test/device-fitness.test.ts` asserts every fixture fits the **floor**,
which is the portability guarantee. The run page states the ceiling at upload
and `verifySystem` carries the same verdict.

🔴 What that does **not** establish is that the kernels produce right numbers on
another GPU. Only running them there does, and `npm run test:browser` is the
lane. **No non-Apple GPU has ever run these kernels**, which is the largest
single unknown in the project and is not a performance question.

The step is also now profiled per kernel (`src/gpu/profile.ts`, reported on
`web/validate.html`, guarded by `test/browser/profile.spec.ts`), and acting on
it took 1L2Y from 208 to 1,213 steps/s and 3HTB from 82 to 151. Two of the
things this list named as next turned out not to be where the time was:
**fewer readbacks** -- there are none in the per-step path, they are all in the
accessors -- and **subgroups**, which were never reached because the costs were
a loop the backend could not schedule around and a dispatch with too few
threads. **Workgroup tuning is still untried** (64 everywhere, valid below
WebGPU's 256-invocation minimum but not necessarily right on a 32-wide NVIDIA
warp or a 16-wide Intel EU), and it is worth more on hardware that is not this
machine.

Still open in Phase 5: workgroup tuning, subgroups, Safari (untestable on macOS
14), accessibility, the tutorial page mirroring the notebook, and GROMACS export
(deviation 10, best-effort).

## 🔴 The things that are not features

These matter more than the phase list, because each one is a limit on how much
any of the above can be trusted.

* ✅ **The thresholds are measured, not provisional.** OpenCL/mixed on an
  NVIDIA L4: force relative RMS 1.43e-05, total energy 0.0136 kJ/mol, NVE drift
  −0.0235 kJ/mol/ns/atom (`docs/measurements/calibration-L4-opencl-mixed.json`).
  Two things remain open inside that. **CUDA is still unmeasured** — every
  precision failed with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, so the basis is
  OpenCL/mixed. The engine has now been measured against
  that basis at the same equilibrated coordinates: **1.10e-06 force relative
  RMS against the 1.43e-05 threshold**, 12.9x inside it and 1.27x the error of
  OpenMM's double-precision CPU platform
  (`docs/measurements/engine-force-error-equilibrated-m3pro.json`).
* 🔴 **NVE drift reproduces only to a factor of ~3.** The same CPU/double
  configuration on nominally the same fixture gives −0.123 on the M3 Pro and
  −0.0443 on the L4 host. Any drift threshold needs a factor-of-several
  tolerance, and tightening one after a passing run would be fitting the
  threshold to the engine.
* **One GPU, one browser.** Everything is measured on an M3 Pro in Chrome. The
  atomic-contention finding in `CLAUDE.md` explicitly says not to act on it
  until it is measured on NVIDIA and on an integrated GPU, and the gather-vs-
  scatter decision for the force kernel waits on that.
* **The Dawn lane cannot run here** (macOS 14), so the second GPU lane is
  unexercised by anyone who has only this machine.
* **The oracle lane has no oracle.** `vendor/openmm-wasm/` does not exist, so
  `npm run test:oracle` self-reports as skipped and 9 tests never run.
* **One unexplained observation**, above: a single browser run had 3HTB's
  LFMiddle step 0.0167 nm from OpenMM while matching the reference to 1.07e-7.
  Never reproduced; fixture checksums now instrument it.
* **`src/gpu/compute.ts` allocates buffers directly**, which `AGENTS.md`
  forbids — every allocation is supposed to go through one place so peak memory
  stays measurable.
* **Three of the four CI jobs are `continue-on-error`**, because none has run on
  a hosted runner. Leaving them soft makes them decorative.
