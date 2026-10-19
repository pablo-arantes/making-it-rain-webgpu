# Where this left off

`AGENTS.md` has the invariants and `CLAUDE.md` the durable findings. This file
has only what is **unfinished**.

## State

**Phase 0 is done.** `npm run check` passes; `npm run test:browser` passes all
five device checks on an Apple M3 Pro through Chrome 155; `npm run bench` and
`npm run report:adapter` work and their output is committed under
`docs/measurements/`. The Python oracle regenerates both committed fixtures.

Phase 0 was meant to be a skeleton plus five working test lanes. It also
settled the engine's numerics, because three candidate schemes went in and one
came out:

- **Forces in 64-bit fixed point** — kept, and exact under contention on the
  device.
- **Positions in double-single** — **abandoned.** Chrome/Metal folds Dekker's
  error term to exactly zero. `docs/PRECISION.md`.
- **Positions in fixed point with residual feedback** — kept. 1.8e-11 nm error
  over 400 ps where plain accumulation drifts 2.3e-5 nm and f32 does not move at
  all.

## Open, in the order they are worth doing

### The acceptance thresholds are provisional
`AGENTS.md` quotes OpenCL **single** precision on an Apple M3 Pro, because this
machine's OpenMM has no mixed-precision GPU platform. Phase 1's gates need the
real thing: `tools/calibrate_thresholds.py` on the **Colab T4**, for 1L2Y, 3HTB
and DHFR. Pablo has agreed to run it. Until then every Phase 1 threshold is a
placeholder with a plausible value, which is the worst kind.

### Atomics may be cheaper than the plan assumed
Measured on Apple: 1 accumulator slot runs at 374 M adds/s, 65,536 slots at
291 M/s — contention is *not* the cost, scattered memory is. `docs/PLAN.md`
budgets a gather-based force kernel with a full neighbour list at 2× the pair
work, partly to avoid contention. That reasoning does not hold on this adapter.
**Do not change the plan on one GPU**: measure NVIDIA and an integrated adapter
first, and note that determinism is a separate argument for gather that this
measurement says nothing about.

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
