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
  else — including a `CustomExternalForce` whose expression is not the one
  restraint the engine implements, and a `NonbondedForce` carrying parameter
  offsets.
- **Bonded forces**, double-precision reference and WGSL kernels: harmonic
  bond, harmonic angle, periodic torsion. Energies match OpenMM's Reference
  platform to between 1.7e-8 and 1.6e-6 relative on both fixtures; forces match
  the reference to between 5.6e-7 and 7.1e-6 relative RMS. Numbers in
  `docs/PRECISION.md`.
- **A plan correction**: an atomic scatter is bit-reproducible, because the
  accumulation is integer. `docs/PLAN.md` had argued for a gather kernel at 2×
  the pair work to keep determinism; it buys no determinism at all. Phase 1's
  remaining kernels are cheaper than planned.

Phase 1 steps still to do, in the plan's order: direct space with exceptions,
PME reciprocal space, constraints, the LFMiddle integrator, the barostat, the
minimiser.

## Open, in the order they are worth doing

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
