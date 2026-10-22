# Engineering invariants

Short on purpose. `CLAUDE.md` is the operational half; `docs/` holds the
measurements these rest on.

## Physics and parity

- **Never change physics silently.** Any deviation from the notebook or from
  OpenMM goes in `docs/DEVIATIONS.md` and is shown to the user when it affects
  their run.
- **Never change reference data or a tolerance to make a failing test pass.**
  The committed numbers in `docs/measurements/` are what OpenMM produced. If our
  answer disagrees, our answer is wrong until proven otherwise.
- **Units are OpenMM's**: nm, ps, amu, kJ/mol, elementary charge, K, bar;
  forces kJ/mol/nm, force constants kJ/mol/nm². The notebook's angstrom,
  nanosecond, femtosecond and molar are converted once, at the UI boundary.
- **Atom ordering is the `System`'s.** Any reordering is a documented mapping,
  never an implicit one.
- **The OpenMM WebAssembly build is a test fixture and a verification tool.** It
  is never a fallback, and it never runs dynamics for a user.

## Numerics

- **Forces accumulate in 64-bit fixed point**, scattered through
  carry-propagating atomics, two 32-bit words, scale 2³²,
  as `src/gpu/kernels/fixed64.wgsl` defines it. Integer addition is exact and
  order-independent. This is not an optimisation: for the 3HTB fixture PME's
  direct space is +1.56e6 kJ/mol and its reciprocal space is −1.73e6, cancelling
  to −1.38e5, so a scheme that rounds while accumulating loses the answer in the
  cancellation, and loses a different amount depending on thread order.
- **Positions are fixed point too, with residual feedback.** The part of a
  displacement that does not fit the format is carried into the next step. Plain
  accumulation drifts when a displacement repeats; measured over 200000
  displacements of 1e-7 nm, plain drifts 2.3e-5 nm and compensated stays at
  1.8e-11 nm. See `docs/PRECISION.md`.
- **Never collapse an absolute position to f32.** Take the difference in fixed
  point, then convert: a pair vector is at most the cutoff, where f32 has room.
  Converting the absolute coordinate throws away the reason it is stored this
  way.
- **No scheme may depend on an f32 rounding surviving the optimiser.**
  Chrome/Metal folds Dekker's `two_sum` error term to exactly zero for runtime
  inputs, because `b - (s - a)` is `b - b` in exact arithmetic. Measured; see
  `docs/PRECISION.md`. Integers cannot be folded, which is why they are used.
- **Both sign branches are tested.** Two of the three numerical bugs found in
  Phase 0 were sign asymmetries that the positive-only case passed.

## GPU

- **Every GPU kernel has a deterministic differential test against an
  independent reference implementation** in `src/reference/`, which never
  imports from `src/gpu/`.
- **A device check is device-in, JSON-out, and lives in `src/`**, so the Dawn
  lane, the browser lane and `npm run bench` run the same code. A check that
  behaved differently in whichever lane was available would be worse than none.
- **Report what was measured, not only whether it passed.** A failing check
  prints its numbers.
- **Establish correctness before performance.** A faster kernel passes the same
  differential test or it does not ship.
- **Route GPU allocations through one place** so peak memory stays measurable.
- **Determinism**: for one device and one seed, a run is reproducible bit for
  bit, and it does not depend on the workgroup configuration. Two things are
  needed for that and only the first is free.

  *Every accumulation **between** threads is exact integer addition*, so its
  result does not depend on the order the adds landed in and an atomic scatter
  is as deterministic as a gather. Introducing a floating-point reduction across
  threads would break this.

  *Every list a kernel walks is in a defined order.* This has been true for
  three different reasons and the current one is not the original. When the pair
  kernel summed its own terms in f32 before one exact add at the end, that sum
  reassociated with the order the terms arrived in -- measured at 1.89e-3
  kJ/mol/nm by feeding one kernel two orderings of the same partition -- so the
  cell list had to be sorted or a trajectory was not reproducible.

  🔴 It no longer does: `accumulate_pair` converts and adds **every pair** in
  fixed point, so a per-atom sum is genuinely order-independent, and
  `validateNeighbours` is the evidence rather than the argument -- the cell-shell
  path combines 27 partial sums an atom and the list path combines 8, different
  groupings of the same terms, agreeing on all 80,247 force components of 3HTB
  bit for bit. Keep the rule anyway: it is what makes the device's cell list
  comparable to the host builder's, which is how that list is validated, and it
  is what gives the neighbour list a defined order. But know which argument is
  load-bearing, because this bullet has twice stated the wrong one.

  Across devices bitwise agreement is not promised, and the reason is recorded.
- **The neighbour list is an optimisation and gives way rather than failing.**
  It is about 4 kB an atom, so it stops fitting one storage binding at around
  33,000 atoms on WebGPU's guaranteed floor. `prepareDirect` then thins the skin
  and, failing that, drops the list and sends every atom through the 27-cell
  shell scan. That path is not a degraded mode kept for the purpose — it is the
  one the differential tests run, and `validateNeighbours` holds the two to
  bit-for-bit agreement — so giving way costs speed and nothing else. A fitness
  report must not call the list's size a problem, or it refuses a system the
  engine can run.
- **The cells are sized by the list's search radius, not by the cutoff.** A
  27-cell shell reaches one cell width, so a radius larger than a cell loses
  pairs — silently, and only on a box big enough to have cells the shell does
  not cover, which no small fixture has.
- **The pair kernel's positions are folded into `[0, box)` before it runs.**
  `cell/wrap` does it once per step, and that precondition is the whole reason
  the kernel's minimum image can be a single conditional shift instead of a
  loop — worth 16x on 3HTB. Any path that encodes the pair pass has to encode
  the fold first; skipping it makes the kernel difference a buffer of zeros,
  which is a wrong answer and not an error. The fold and the cell list's own
  bucketing share one function, so an atom's home cell and its pair vectors
  cannot disagree.
- **Measure performance on the largest fixture, not the smallest.** A step cost
  4.8 ms on 1L2Y and 629 ms on 3HTB for 11x the atoms, and the two defects in
  that gap were invisible on 1L2Y: one fix moved it 2% and moved 3HTB 18x.
- **Validate shapes, dtypes and byte sizes at public boundaries**, and use a
  WebGPU error scope around every dispatch: a validation error otherwise appears
  as a console warning and a silently wrong result, which is the one failure a
  differential test cannot tell from a bad kernel.
- **Target Chrome first** while keeping standards-compliant WebGPU and WGSL.

## Acceptance thresholds

The engine's target is OpenMM's mixed-precision GPU platform, not its
double-precision Reference platform. Measured, on an equilibrated 1L2Y — each
run equilibrates 10 ps first, because at a fixture's raw coordinates the
relative force error reports on the input structure rather than on the platform:

| | force rel. RMS vs Reference | total energy | NVE drift at 2 fs |
| --- | ---: | ---: | ---: |
| **OpenMM OpenCL mixed, NVIDIA L4** | **1.43e-05** | **+0.0136 kJ/mol** | **−0.0235 kJ/mol/ns/atom** |
| OpenMM OpenCL single, NVIDIA L4 | 1.43e-05 | +0.0149 kJ/mol | +0.0479 kJ/mol/ns/atom |
| OpenMM CPU double, L4 host | 1.03e-06 | −0.0071 kJ/mol | −0.0443 kJ/mol/ns/atom |
| OpenMM OpenCL single, Apple M3 Pro | 1.05e-05 | −0.0019 kJ/mol | −0.168 kJ/mol/ns/atom |
| OpenMM CPU double, Apple M3 Pro | 8.6e-07 | −0.0046 kJ/mol | −0.123 kJ/mol/ns/atom |

The bold row is the basis. From
`docs/measurements/calibration-L4-opencl-mixed.json`; the M3 Pro rows are from
`docs/measurements/calibration-M3Pro-opencl.json` and are kept because
agreement across two unrelated devices is what makes the force figure a floor
rather than one machine's number.

**~1e-5 relative force error is the floor for a legitimate single-precision
implementation.** Two independent devices give 1.05e-05 and 1.43e-05, and their
double-precision CPU rows give 8.6e-07 and 1.03e-06 — so this quantity
reproduces to about 20% across hosts and can carry a tight threshold.

🔴 **NVE drift cannot.** The *same* configuration — CPU, double precision,
nominally the same fixture (2349 and 2355 atoms) — gives −0.123 and −0.0443
kJ/mol/ns/atom on the two hosts, a factor of 2.8. Over a picosecond the total
energy fluctuates by about as much as the trend, so the slope is a fit to noise
as much as to drift. A drift threshold must therefore carry a factor-of-several
tolerance; a tight one would fail a correct implementation, and tightening it
after a passing run would be fitting the threshold to the engine.

Two things this set does **not** establish, and neither is a reason to loosen
it:

* **CUDA is unmeasured.** All three CUDA precisions failed on the Colab host
  with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, so the basis is OpenCL/mixed.
  Both are genuine mixed precision, but OpenMM's CUDA and OpenCL platforms are
  separate implementations and may not agree to better than the spread above.

### Where the engine sits against it

Measured at the configuration the thresholds are defined at — the calibration's
own equilibration, saved with `--save-equilibrated` so every engine sees the
same coordinates, because equilibration is stochastic and re-running it gives
an incomparable structure. 1L2Y, 2373 particles,
`docs/measurements/engine-force-error-equilibrated-m3pro.json`:

| | force rel. RMS | total energy error |
| --- | ---: | ---: |
| OpenMM OpenCL single | 1.42e-05 | 0.0044 kJ/mol |
| **this engine, WebGPU** | **1.10e-06** | **0.0084 kJ/mol** |
| OpenMM CPU double | 8.67e-07 | 0.0023 kJ/mol |

🔴 Re-measure this with `npm run report:engine-error` after touching a kernel,
and do not take a passing threshold as the answer. It was produced by hand once,
then four kernel changes went past it — the shell split, the loop-free minimum
image, the position fold and the neighbour list — with no way to tell whether it
still described what ships. Re-measured afterwards it was **bit-identical to all
17 digits**, forces and energy both, which said those four changes were *exact*
and not merely inside tolerance. A threshold could not have told the difference.

PME's triclinic conversion then moved it, at the f32 ulp: the kernels compute a
fractional coordinate as `r · B⁻¹` where they divided by an edge length, and
`r * (1/L)` is not `r / L` in f32. 1.1046e-06 to **1.0959e-06**, which is 0.8%
*better* and still 13× inside the basis. Accepted rather than branched around,
because the reference computes `r · B⁻¹` too — the kernel and the thing it is
tested against now do the same arithmetic, which is worth more than preserving a
digit.

The engine is **12.9× inside** the single-precision basis on forces, and sits
nearer OpenMM's *double*-precision CPU platform (1.27× its error) than its
single-precision GPU one. That is the designed consequence of 64-bit
fixed-point positions and exact force accumulation: the only single-precision
step is the per-pair evaluation itself.

🔴 **The asymmetry is real and is not a defect.** The total energy is about
twice as far out as OpenMM's single-precision platform even though the forces
are thirteen times better. The total is a difference of two large cancelling
PME terms — about +122,000 and −154,000 kJ/mol here — so its relative error is
amplified by the cancellation, not by the accumulator. Do not "fix" the energy
by changing the force accumulator.

Both sides use the identical formula, `sqrt(Σ(a−b)² / Σb²)` over all 3N
components, in `src/gpu/verify.ts` and `tools/calibrate_thresholds.py`. Each
engine is compared against *its own* f64 reference, which is only legitimate
because the two references agree: at these coordinates ours and OpenMM's
Reference differ by 5.2e-12 relative, matching the 9.2e-12 that
`test/crossfeed.test.ts` finds on a different system.

`tools/calibrate_thresholds.py` produces this table, and raising a threshold is
a reviewed change with a reason, never a convenience.
