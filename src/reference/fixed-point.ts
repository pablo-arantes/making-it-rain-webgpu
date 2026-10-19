/**
 * The exact reference for the fixed-point accumulation the GPU does.
 *
 * `src/reference/` holds independent implementations that the WGSL kernels are
 * differentially tested against.  They are written for clarity and exactness,
 * never for speed, and they never import from `src/gpu/`.
 *
 * Here "exact" is literal: the accumulation is integer arithmetic, so the
 * reference uses BigInt and the GPU must agree with it bit for bit.  Any
 * disagreement is a bug in the kernel, not a tolerance to be widened.
 */

/** 2^32, the fixed-point scale.  OpenMM's GPU platforms use the same one. */
export const FIXED_SCALE = 4294967296n;

/** The representable range, in kJ/mol/nm. */
export const FIXED_MAX = 2 ** 31;

/**
 * Round a non-negative value to the nearest integer, ties to the even one.
 *
 * WGSL's `round` resolves ties to even and JavaScript's `Math.round` resolves
 * them upward, and the two sides have to agree bit for bit.
 */
function roundTiesToEven(x: number): number {
  const whole = Math.floor(x);
  const frac = x - whole;
  if (frac > 0.5) return whole + 1;
  if (frac < 0.5) return whole;
  return whole % 2 === 0 ? whole : whole + 1;
}

/**
 * Split an f32 into the two 32-bit words of its fixed-point form.
 *
 * Mirrors `to_fixed` in src/gpu/kernels/fixed64.wgsl step for step, so the GPU
 * and this reference produce identical words for identical input.
 *
 * The magnitude is decomposed and the result negated, rather than flooring a
 * signed value: for a small negative x, `x - floor(x)` is `1 + x`, and an f32
 * near 1 cannot keep the low bits of x, which costs 2.2e-8 against 2.3e-10 for
 * the same magnitude positive.
 *
 * The fractional word rounds to nearest rather than truncating, because
 * truncation is biased toward zero and a bias accumulates: 200000
 * displacements of 1e-7 nm drifted 2.3e-5 nm low, growing linearly with
 * simulation length.
 *
 * `value` must already be an f32 (use `Math.fround`), because the kernel sees
 * an f32 and the comparison is exact.
 *
 * Returns the low word unsigned and the high word as its signed i32 reading,
 * which is how the GPU's two buffers read back.
 */
export function fixedWords(value: number): { lo: number; hi: number } {
  if (!Number.isFinite(value)) throw new RangeError(`not finite: ${value}`);
  if (Math.abs(value) >= FIXED_MAX) {
    throw new RangeError(`${value} kJ/mol/nm is outside the fixed-point range`);
  }
  const magnitude = Math.abs(value);
  const whole = Math.floor(magnitude);
  // Exact: both operands are non-negative and the difference is representable.
  const frac = magnitude - whole;
  const lo = roundTiesToEven(frac * 2 ** 32);
  if (value >= 0) return { lo, hi: whole };
  // Two's complement of the 64-bit value, as `neg64` computes it.
  return lo === 0 ? { lo: 0, hi: -whole } : { lo: 2 ** 32 - lo, hi: -whole - 1 };
}

/** The 64-bit fixed-point integer for an f32, at the format's own resolution. */
export function toFixed(value: number): bigint {
  const { lo, hi } = fixedWords(value);
  return BigInt(hi) * FIXED_SCALE + BigInt(lo);
}

/**
 * Recover the value a fixed-point integer represents, in f32, exactly as
 * `from_fixed` in src/gpu/kernels/fixed64.wgsl computes it.
 *
 * Mirrors the kernel operation for operation, including the order of the
 * rounding, because the integrator's residual feeds back through this and any
 * divergence between device and reference compounds over a run.  Use
 * `fromFixed` for an exact f64 reading; use this one when comparing against a
 * device.
 */
export function fromFixedF32(fixed: bigint): number {
  const negative = fixed < 0n;
  const magnitude = negative ? -fixed : fixed;
  const hi = Math.fround(Number(magnitude >> 32n));
  const lo = Math.fround(Math.fround(Number(magnitude & 0xffffffffn)) * 2 ** -32);
  const value = Math.fround(hi + lo);
  return negative ? -value : value;
}

/** Recover the value a fixed-point integer represents, as an f64. */
export function fromFixed(fixed: bigint): number {
  // Done in two parts so no intermediate exceeds 2^53.
  const whole = fixed / FIXED_SCALE;
  const frac = fixed - whole * FIXED_SCALE;
  return Number(whole) + Number(frac) / 2 ** 32;
}

/** Reassemble the two words the GPU wrote.  `lo` is unsigned, `hi` signed. */
export function fromWords(lo: number, hi: number): bigint {
  return BigInt(hi) * FIXED_SCALE + BigInt(lo >>> 0);
}

/**
 * The sum of a list of f32 values in fixed point, computed exactly.
 *
 * Each value is converted at the format's 2^-32 resolution, exactly as the
 * kernel converts it, and the additions are then exact integer additions. This
 * is what the GPU must produce, in any thread order, with no tolerance.
 */
export function exactFixedSum(values: ArrayLike<number>): bigint {
  let total = 0n;
  for (let i = 0; i < values.length; i++) total += toFixed(values[i] as number);
  return total;
}

/**
 * The same sum accumulated in f32, left to right.  Not used by the engine --
 * it exists so a test can show what the fixed-point scheme is buying.
 */
export function f32Sum(values: ArrayLike<number>): number {
  let total = 0;
  for (let i = 0; i < values.length; i++) {
    total = Math.fround(total + (values[i] as number));
  }
  return total;
}
