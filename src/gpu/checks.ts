/**
 * Device checks for the two precision decisions in `AGENTS.md`.
 *
 * These live in `src/` rather than `test/` because three lanes run exactly the
 * same code: Vitest through Dawn, Playwright through a real browser, and
 * `npm run bench`.  A check that behaved differently in the lane that happened
 * to be available would be worse than no check.
 *
 * Every check is device-in, JSON-out, and says what it measured rather than
 * only whether it passed.
 */

import { describeAdapter, estimateMaxAtoms, type AdapterReport } from "./device.js";
import { accumulateFixedPoint, integratePosition } from "./precision.js";
import {
  dsAccumulate,
  dsToF64,
  f32Accumulate,
  f64Accumulate,
} from "../reference/double-single.js";
import { exactFixedSum, fromFixed } from "../reference/fixed-point.js";
import { fft3d, planFft } from "../reference/fft.js";
import { storageBuffer, readBuffer, uploadBuffer } from "./compute.js";
import { transform3d } from "./pme.js";
import { integrate, integrateCompensated } from "../reference/position.js";

export interface CheckResult {
  readonly name: string;
  readonly ok: boolean;
  readonly claim: string;
  readonly detail: Readonly<Record<string, unknown>>;
}

/** A deterministic PRNG, so a failing check is a reproducible one. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Fixed-point accumulation is exact, whatever order the threads run in.
 *
 * 65536 force contributions are accumulated into 16 slots, so about 4096
 * invocations contend for each one.  The reference is BigInt, so the
 * comparison has no tolerance: equal, or the kernel is wrong.
 */
export async function checkFixedPointExactness(
  device: GPUDevice,
): Promise<CheckResult> {
  const random = mulberry32(0x5eed);
  const values = new Float32Array(65536);
  for (let i = 0; i < values.length; i++) {
    // Forces spanning the range a real system produces, both signs.
    const magnitude = 10 ** (random() * 5 - 1); // 0.1 .. 10000 kJ/mol/nm
    values[i] = Math.fround((random() < 0.5 ? -1 : 1) * magnitude);
  }
  const slotCount = 16;
  const gpu = await accumulateFixedPoint(device, values, slotCount);

  const expected: bigint[] = [];
  for (let slot = 0; slot < slotCount; slot++) {
    const mine: number[] = [];
    for (let i = slot; i < values.length; i += slotCount) mine.push(values[i] as number);
    expected.push(exactFixedSum(mine));
  }
  const mismatches = expected
    .map((want, slot) => ({ slot, want, got: gpu[slot] as bigint }))
    .filter((r) => r.want !== r.got);

  return {
    name: "fixed-point accumulation is exact under contention",
    ok: mismatches.length === 0,
    claim:
      "65536 signed force contributions accumulated into 16 slots must equal the " +
      "exact BigInt sum, with no tolerance, whatever order the GPU ran them in",
    detail: {
      values: values.length,
      slots: slotCount,
      contributionsPerSlot: values.length / slotCount,
      mismatches: mismatches.length,
      firstMismatch: mismatches[0]
        ? {
            slot: mismatches[0].slot,
            want: mismatches[0].want.toString(),
            got: mismatches[0].got.toString(),
            differenceKJPerMolNm:
              fromFixed(mismatches[0].got) - fromFixed(mismatches[0].want),
          }
        : null,
      slot0SumKJPerMolNm: fromFixed(expected[0] as bigint),
    },
  };
}

/**
 * The hand-rolled carry out of the low word is correct.
 *
 * There are no 64-bit atomics in WebGPU, so `accumulate()` adds the low words
 * and detects the wrap from the value `atomicAdd` returns.  This check forces
 * that path hard: every contribution has a large fractional part, so the low
 * word overflows thousands of times into a single slot, and the signs are
 * mixed so carries and borrows interleave.
 */
export async function checkFixedPointCarry(device: GPUDevice): Promise<CheckResult> {
  const random = mulberry32(0xc0ffee);
  const values = new Float32Array(16384);
  for (let i = 0; i < values.length; i++) {
    // Fractions near 1 maximise low-word overflow; the integer part stays
    // small so the total remains well inside the representable range.
    const frac = 0.9 + random() * 0.0999;
    values[i] = Math.fround((random() < 0.5 ? -1 : 1) * frac);
  }
  const gpu = await accumulateFixedPoint(device, values, 1);
  const expected = exactFixedSum(values);
  const got = gpu[0] as bigint;
  // How many times the low word wrapped, which is exactly how many carries the
  // kernel had to get right.  64-bit two's-complement addition is unsigned
  // addition of the low words with the carry going into the high word, so
  // counting wraps of the unsigned low-word sum counts carries exactly.
  let wraps = 0n;
  let low = 0n;
  for (let i = 0; i < values.length; i++) {
    low += exactFixedSum([values[i] as number]) & 0xffffffffn;
    if (low >= 0x100000000n) {
      low -= 0x100000000n;
      wraps += 1n;
    }
  }
  return {
    name: "the 64-bit carry emulation is correct",
    ok: expected === got,
    claim:
      "16384 contributions with fractional parts near 1, both signs, into one " +
      "slot: the emulated carry must reproduce exact 64-bit addition",
    detail: {
      values: values.length,
      lowWordWraps: Number(wraps),
      want: expected.toString(),
      got: got.toString(),
      differenceKJPerMolNm: fromFixed(got) - fromFixed(expected),
    },
  };
}

/**
 * Positions in fixed point move where plain f32 positions do not, and the
 * compensated integrator does not drift while doing it.
 *
 * The displacement is 1e-7 nm, which is below half an ulp of a 6.5 nm f32
 * coordinate (that ulp is 4.77e-7 nm), so an f32 accumulator does not merely
 * drift -- it stops moving entirely.  The 3HTB box is 6.52 nm, and the
 * notebook's 1e-6 constraint tolerance on a 0.1 nm X-H bond is 1e-7 nm, so this
 * is the regime the constraint solver actually works in.
 *
 * Repeating one displacement is the hardest case for a fixed-point integrator
 * and the easiest to overlook: the conversion error is identical every step, so
 * it adds up instead of cancelling.  Both accumulators are therefore measured,
 * and the device must agree with the reference bit for bit on both.
 */
export async function checkPositionStagnation(
  device: GPUDevice,
): Promise<CheckResult> {
  const startNm = 6.5;
  const count = 200_000;
  const displacements = new Float32Array(count).fill(Math.fround(1e-7));

  const gpu = await integratePosition(device, startNm, displacements);
  const plain = integrate(startNm, displacements);
  const compensated = integrateCompensated(startNm, displacements);
  const exactNm = f64Accumulate(startNm, displacements);

  const plainErrorNm = Math.abs(fromFixed(plain) - exactNm);
  const compensatedErrorNm = Math.abs(fromFixed(compensated.position) - exactNm);
  const singleErrorNm = Math.abs(gpu.singleNm - exactNm);

  return {
    name: "fixed-point positions move where f32 stagnates, without drifting",
    ok:
      gpu.plain === plain &&
      gpu.compensated === compensated.position &&
      compensatedErrorNm < plainErrorNm / 100 &&
      compensatedErrorNm < 1e-7 &&
      singleErrorNm > 0.01,
    claim:
      "200000 displacements of 1e-7 nm onto a 6.5 nm coordinate: both device " +
      "accumulators must equal their references bit for bit, the compensated one " +
      "must stay inside 1e-7 nm of the exact answer, and f32 must visibly fail",
    detail: {
      startNm,
      displacements: count,
      displacementNm: 1e-7,
      f32UlpAtStartNm: 2 ** -21,
      exactNm,
      plainNm: gpu.plainNm,
      plainErrorNm,
      plainMatchesReference: gpu.plain === plain,
      compensatedNm: gpu.compensatedNm,
      compensatedErrorNm,
      compensatedMatchesReference: gpu.compensated === compensated.position,
      compensationFactor: plainErrorNm / Math.max(compensatedErrorNm, Number.MIN_VALUE),
      outstandingResidualNm: gpu.residualNm,
      f32Nm: gpu.singleNm,
      f32ErrorNm: singleErrorNm,
      f32MovedAtAll: gpu.singleNm !== startNm,
    },
  };
}

/**
 * The same on a realistic trajectory of displacements.
 *
 * A 2 fs step moves a hydrogen a few times 1e-4 nm, so this walks 200000 steps
 * of random +/-5e-4 nm -- 400 ps of one coordinate's history.  Here the
 * conversion errors decorrelate, so plain accumulation random-walks rather than
 * drifting and both accumulators should do well; what the check is really for
 * is that the device still matches the reference exactly over 200000
 * dependent integer operations.
 */
export async function checkPositionRandomWalk(
  device: GPUDevice,
): Promise<CheckResult> {
  const random = mulberry32(0xbeef);
  const startNm = 6.5;
  const count = 200_000;
  const displacements = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    displacements[i] = Math.fround((random() * 2 - 1) * 5e-4);
  }

  const gpu = await integratePosition(device, startNm, displacements);
  const plain = integrate(startNm, displacements);
  const compensated = integrateCompensated(startNm, displacements);
  const exactNm = f64Accumulate(startNm, displacements);

  return {
    name: "fixed-point positions track a realistic displacement history",
    ok: gpu.plain === plain && gpu.compensated === compensated.position,
    claim:
      "200000 displacements of +/-5e-4 nm, the scale of a 2 fs step: both device " +
      "accumulators must equal their references bit for bit",
    detail: {
      startNm,
      steps: count,
      exactNm,
      plainNm: gpu.plainNm,
      plainErrorNm: fromFixed(plain) - exactNm,
      plainMatchesReference: gpu.plain === plain,
      compensatedNm: gpu.compensatedNm,
      compensatedErrorNm: fromFixed(compensated.position) - exactNm,
      compensatedMatchesReference: gpu.compensated === compensated.position,
      outstandingResidualNm: gpu.residualNm,
      f32Nm: gpu.singleNm,
      f32ErrorNm: gpu.singleNm - exactNm,
    },
  };
}

/**
 * What this device does with Dekker error-free arithmetic.
 *
 * A diagnostic, not a gate: it passes whatever the device does, and records
 * which.  Double-single was the first plan for positions and is a standard
 * technique, but on Chrome/Metal the error term of `two_sum` comes back as
 * exactly zero for runtime inputs, because the backend simplifies
 * `b - (s - a)` into `b - b`.  Positions are integers because of that
 * measurement, so the measurement travels with the code: if some backend ever
 * computes the sequence correctly, this says so instead of leaving the
 * decision resting on a memory.  See docs/PRECISION.md.
 */
export async function checkDekkerDiagnostic(device: GPUDevice): Promise<CheckResult> {
  const startNm = 6.5;
  const count = 200_000;
  const displacements = new Float32Array(count).fill(Math.fround(1e-7));

  const gpu = await integratePosition(device, startNm, displacements);
  const expectedNm = f64Accumulate(startNm, displacements);
  const hostDekker = dsToF64(dsAccumulate(startNm, displacements));
  const hostF32 = f32Accumulate(startNm, displacements);
  const deviceError = Math.abs(gpu.dekker.valueNm - expectedNm);
  const honoursErrorFreeArithmetic = deviceError < 1e-6;

  return {
    name: honoursErrorFreeArithmetic
      ? "this device computes Dekker error-free arithmetic correctly"
      : "this device folds Dekker error-free arithmetic away (expected)",
    ok: true, // diagnostic
    claim:
      "a report, not a requirement: whether the backend preserves the rounding " +
      "that double-single depends on. Positions use fixed-point integers either way",
    detail: {
      honoursErrorFreeArithmetic,
      deviceDekkerNm: gpu.dekker.valueNm,
      deviceDekkerHi: gpu.dekker.hi,
      deviceDekkerLo: gpu.dekker.lo,
      deviceDekkerErrorNm: deviceError,
      hostDekkerNm: hostDekker,
      hostDekkerErrorNm: Math.abs(hostDekker - expectedNm),
      hostF32Nm: hostF32,
      expectedNm,
      note: honoursErrorFreeArithmetic
        ? "double-single would work here; the engine still uses integers, which " +
          "work everywhere"
        : "the low word stayed at zero, so `b - (s - a)` was folded to zero",
    },
  };
}

/**
 * The mixed-radix FFT kernel against `src/reference/fft.ts`.
 *
 * Needs no fixture, so it runs in every lane alongside the precision checks.
 * The grids exercise each radix on its own and mixed together, and a non-cubic
 * shape, because the index arithmetic that maps a line index to a flat offset
 * is shared by all three axes and is the part most likely to be wrong.
 *
 * f32 against f64 over a transform whose intermediate values grow like the
 * grid size, so a few times 1e-6 relative is what a correct kernel looks like
 * here, not 1e-7.
 */
export async function checkFft(device: GPUDevice): Promise<CheckResult> {
  const shapes: ReadonlyArray<readonly [number, number, number]> = [
    [8, 8, 8],
    [9, 9, 9],
    [25, 25, 25],
    [54, 4, 6],
    [7, 5, 3],
  ];
  const random = mulberry32(0x7f7);
  const results: Array<Record<string, unknown>> = [];
  let worst = 0;

  for (const shape of shapes) {
    const [nx, ny, nz] = shape;
    const total = nx * ny * nz;
    const original = new Float32Array(total * 2);
    for (let i = 0; i < original.length; i++) {
      original[i] = Math.fround(random() * 2 - 1);
    }

    const a = uploadBuffer(device, {
      label: "fft check a",
      data: original,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const b = storageBuffer(device, "fft check b", total * 2 * 4);
    let deviceResult: Float32Array;
    try {
      const plans = [nz, ny, nx].map((length) => planFft(length));
      const held = await transform3d(device, a, b, shape, -1, plans);
      deviceResult = new Float32Array(await readBuffer(device, held, total * 2 * 4));
    } finally {
      a.destroy();
      b.destroy();
    }

    const expected = Float64Array.from(original);
    fft3d(expected, shape, -1);
    let sumSquaredError = 0;
    let sumSquared = 0;
    for (let i = 0; i < expected.length; i++) {
      const error = (deviceResult[i] as number) - (expected[i] as number);
      sumSquaredError += error * error;
      sumSquared += (expected[i] as number) ** 2;
    }
    const relative = Math.sqrt(sumSquaredError / sumSquared);
    worst = Math.max(worst, relative);
    results.push({
      shape: `${nx}x${ny}x${nz}`,
      radices: [nz, ny, nx].map((n) => planFft(n).stages.map((s) => s.radix).join("x")),
      relativeRms: relative,
    });
  }

  return {
    name: "the mixed-radix FFT matches the reference transform",
    ok: worst < 1e-5,
    claim:
      "forward 3D transforms of five grid shapes, covering radices 2, 3, 5 and 7 " +
      "separately and mixed, must match the f64 reference to better than 1e-5 " +
      "relative RMS",
    detail: { worstRelativeRms: worst, shapes: results },
  };
}

export interface DeviceCheckReport {
  readonly adapter: AdapterReport;
  readonly sizeEstimate: ReturnType<typeof estimateMaxAtoms>;
  readonly checks: readonly CheckResult[];
  readonly ok: boolean;
}

/** Run every check and describe the device that ran them. */
export async function runDeviceChecks(
  adapter: GPUAdapter,
  device: GPUDevice,
): Promise<DeviceCheckReport> {
  const checks: CheckResult[] = [];
  for (const run of [
    checkFixedPointExactness,
    checkFixedPointCarry,
    checkPositionStagnation,
    checkPositionRandomWalk,
    checkDekkerDiagnostic,
    checkFft,
  ]) {
    try {
      checks.push(await run(device));
    } catch (error) {
      checks.push({
        name: run.name,
        ok: false,
        claim: "the check must run at all",
        detail: { error: String(error) },
      });
    }
  }
  return {
    adapter: describeAdapter(adapter),
    sizeEstimate: estimateMaxAtoms(device.limits),
    checks,
    ok: checks.every((c) => c.ok),
  };
}
