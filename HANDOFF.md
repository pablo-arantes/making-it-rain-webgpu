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
  tolerance they were asked for.
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

Phase 1 steps still to do: the **integrator kernel** (including
centre-of-mass removal, which needs a reduction), the **step loop** on resident
buffers, the **barostat**, and the **minimiser**. After those, Phase 1's gates:
NVE drift at 2 fs, and a measured ns/day.

Everything the physics needs exists in `src/reference/`, verified against
OpenMM to 5.7e-9 nm a step, and every force plus the constraints now also run
on the GPU. What remains is the integrator kernel and the loop.

## Open, in the order they are worth doing

### The FFT is correct but not yet tuned
Radices 2, 3, 5 and 7, with a generic R² inner product per butterfly rather
than specialised butterflies — at most 49 terms, since the radix is at most 7.
Specialised radix-2/3/4/5 butterflies and a radix-4 or radix-8 path for
power-of-two sizes are the obvious next step, and the differential test in
`src/gpu/checks.ts` is what they would be held to. **Measure before writing
them**: the per-step cost of PME against everything else is not known yet,
because there is no step loop to measure.

### RATTLE on the GPU still goes through SHAKE for water
The reference solves a rigid triangle's velocity constraints with SETTLE's exact
3x3 solve; the kernel currently sends every cluster, triangles included, through
the iterative path. Both converge to the same condition and the difference is
inside the tolerance, but it is an avoidable discrepancy and the exact solve is
about twenty lines of WGSL.

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
