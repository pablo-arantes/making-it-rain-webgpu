/**
 * Host side of the two precision primitives the whole engine rests on.
 *
 * These exist in Phase 0 rather than Phase 1 on purpose.  `AGENTS.md` commits
 * the engine to fixed-point force accumulation and double-single positions
 * before a single force kernel is written, and both commitments depend on
 * properties of the device -- that 32-bit atomics wrap the way the carry trick
 * needs, and that the backend does not reassociate the error-free arithmetic
 * double-single is built from.  Neither can be settled by reading the WGSL
 * specification, so they are measured here, first.
 */

import { dispatch, readBuffer, storageBuffer, uploadBuffer } from "./compute.js";
import { fromFixed, fromWords } from "../reference/fixed-point.js";
import { FIXED_POINT_ACCUMULATE, POSITION } from "./kernels.js";

/**
 * Accumulate `values` into `slotCount` slots on the GPU, in 64-bit fixed point.
 *
 * Value `i` goes to slot `i % slotCount`, so a small `slotCount` makes many
 * invocations contend for the same accumulator -- which is the case worth
 * testing, because it is the case a force kernel is.
 */
export async function accumulateFixedPoint(
  device: GPUDevice,
  values: Float32Array,
  slotCount: number,
): Promise<bigint[]> {
  if (slotCount <= 0) throw new RangeError("slotCount must be positive");
  const valueBuffer = uploadBuffer(device, {
    label: "fixed-point values",
    data: values,
    usage: GPUBufferUsage.STORAGE,
  });
  const lo = storageBuffer(device, "fixed-point acc_lo", slotCount * 4);
  const hi = storageBuffer(device, "fixed-point acc_hi", slotCount * 4);
  // Both accumulators start at zero.  `createBuffer` zero-fills, but writing it
  // explicitly keeps the kernel's contract visible at the call site.
  device.queue.writeBuffer(lo, 0, new Uint32Array(slotCount));
  device.queue.writeBuffer(hi, 0, new Int32Array(slotCount));
  const params = uploadBuffer(device, {
    label: "fixed-point params",
    data: new Uint32Array([values.length, slotCount]),
    usage: GPUBufferUsage.UNIFORM,
  });

  try {
    await dispatch(device, {
      label: "fixed-point accumulate",
      code: FIXED_POINT_ACCUMULATE,
      bindings: [
        { buffer: valueBuffer },
        { buffer: lo },
        { buffer: hi },
        { buffer: params },
      ],
      workgroups: [Math.ceil(values.length / 64)],
    });
    const loWords = new Uint32Array(await readBuffer(device, lo, slotCount * 4));
    const hiWords = new Int32Array(await readBuffer(device, hi, slotCount * 4));
    return Array.from({ length: slotCount }, (_, i) =>
      fromWords(loWords[i] as number, hiWords[i] as number),
    );
  } finally {
    for (const b of [valueBuffer, lo, hi, params]) b.destroy();
  }
}

export interface PositionResult {
  /** Plain fixed-point accumulation: convert each displacement and add it. */
  readonly plain: bigint;
  readonly plainNm: number;
  readonly plainWords: { readonly lo: number; readonly hi: number };
  /** Fixed point with the unfitted part of each displacement carried forward. */
  readonly compensated: bigint;
  readonly compensatedNm: number;
  readonly compensatedWords: { readonly lo: number; readonly hi: number };
  /** The residual still outstanding when the run ended, in nm. */
  readonly residualNm: number;
  /** The same accumulation in plain f32, from the same dispatch. */
  readonly singleNm: number;
  /** What Dekker double-single produced on this device: a diagnostic. */
  readonly dekker: { readonly hi: number; readonly lo: number; readonly valueNm: number };
}

/**
 * Integrate one coordinate from `startNm` by a list of f32 displacements, on
 * the device, in all three schemes at once.
 *
 * Doing all three in one dispatch is the point: the error of any of them
 * depends on the device, the driver, and the magnitude of `startNm`, so they
 * have to be measured against each other rather than separately.
 */
export async function integratePosition(
  device: GPUDevice,
  startNm: number,
  displacements: Float32Array,
): Promise<PositionResult> {
  const incrementBuffer = uploadBuffer(device, {
    label: "position displacements",
    data: displacements,
    usage: GPUBufferUsage.STORAGE,
  });
  const out = storageBuffer(device, "position results", 8 * 4);
  const params = uploadBuffer(device, {
    label: "position params",
    data: new Uint32Array([
      displacements.length,
      // `startNm` travels as an f32 bit pattern in a u32 slot so the uniform
      // stays a single 8-byte struct with no padding surprises.
      new Uint32Array(new Float32Array([startNm]).buffer)[0] as number,
    ]),
    usage: GPUBufferUsage.UNIFORM,
  });

  try {
    await dispatch(device, {
      label: "position integrate",
      code: POSITION,
      bindings: [{ buffer: incrementBuffer }, { buffer: out }, { buffer: params }],
      workgroups: [1],
    });
    const raw = await readBuffer(device, out, 32);
    const words = new Uint32Array(raw);
    const floats = new Float32Array(raw);
    const plainWords = { lo: words[0] as number, hi: (words[1] as number) | 0 };
    const compensatedWords = { lo: words[2] as number, hi: (words[3] as number) | 0 };
    const plain = fromWords(plainWords.lo, plainWords.hi);
    const compensated = fromWords(compensatedWords.lo, compensatedWords.hi);
    return {
      plain,
      plainNm: fromFixed(plain),
      plainWords,
      compensated,
      compensatedNm: fromFixed(compensated),
      compensatedWords,
      residualNm: floats[7] as number,
      singleNm: floats[4] as number,
      dekker: {
        hi: floats[5] as number,
        lo: floats[6] as number,
        valueNm: (floats[5] as number) + (floats[6] as number),
      },
    };
  } finally {
    for (const b of [incrementBuffer, out, params]) b.destroy();
  }
}
