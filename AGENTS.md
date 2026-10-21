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

  *Every list a kernel walks is in a defined order.* A thread may still sum its
  own terms in f32 before the one exact add at the end -- `nonbonded-pairs.wgsl`
  does, and so does OpenMM -- and that sum reassociates with the order the terms
  come in. So the cell list is sorted rather than left as the atomics produced
  it. Measured, by feeding one kernel two orderings of the same partition:
  1.89e-3 kJ/mol/nm, about 2e-6 of the force RMS. This bullet previously claimed
  the exact accumulator made ordering irrelevant; it does not, and the claim
  cost nothing only because the test that found it was written to check it.

  Across devices bitwise agreement is not promised, and the reason is recorded.
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
* **CUDA is unmeasured** (above), so the basis is OpenCL/mixed.

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
