/**
 * Dekker error-free arithmetic, kept as a diagnostic reference and used by
 * nothing in the engine.
 *
 * This was the first plan for positions: two f32 words holding a value as
 * their exact unevaluated sum, giving about 48 bits of mantissa.  It is a
 * standard technique and it is correct on hardware that computes what the
 * source says.
 *
 * Chrome/Metal does not.  With runtime inputs, `two_sum`'s error term comes
 * back as exactly 0 on an Apple M3 Pro, because the backend simplifies
 * `b - (s - a)` into `b - b`.  That identity holds in exact arithmetic and is
 * false in floating point, and it destroys the only thing the expression
 * exists to compute.  Measured in docs/PRECISION.md; positions use fixed-point
 * integers instead (src/reference/position.ts).
 *
 * This file stays because the check in src/gpu/checks.ts reports what each
 * device does with the sequence, and it needs the correct answer to compare
 * against.  If some backend ever computes it properly, we will see that rather
 * than assume it.
 */

const f = Math.fround;

export type DoubleSingle = readonly [hi: number, lo: number];

/** The exact sum of two f32 values as (sum, error). */
export function twoSum(a: number, b: number): DoubleSingle {
  const s = f(a + b);
  const bb = f(s - a);
  const err = f(f(a - f(s - bb)) + f(b - bb));
  return [s, err];
}

export function quickTwoSum(a: number, b: number): DoubleSingle {
  const s = f(a + b);
  return [s, f(b - f(s - a))];
}

export const dsFromF32 = (a: number): DoubleSingle => [f(a), 0];

export const dsToF32 = (a: DoubleSingle): number => f(a[0] + a[1]);

/** The value a double-single pair represents, without collapsing it to f32. */
export const dsToF64 = (a: DoubleSingle): number => a[0] + a[1];

export function dsAdd(a: DoubleSingle, b: DoubleSingle): DoubleSingle {
  const s = twoSum(a[0], b[0]);
  const t = twoSum(a[1], b[1]);
  const e = f(s[1] + t[0]);
  const hi = quickTwoSum(s[0], e);
  return quickTwoSum(hi[0], f(hi[1] + t[1]));
}

export function dsAddF32(a: DoubleSingle, b: number): DoubleSingle {
  const s = twoSum(a[0], b);
  return quickTwoSum(s[0], f(s[1] + a[1]));
}

/** Accumulate f32 increments onto a start value, in double-single. */
export function dsAccumulate(
  start: number,
  increments: ArrayLike<number>,
): DoubleSingle {
  let acc = dsFromF32(start);
  for (let i = 0; i < increments.length; i++) {
    acc = dsAddF32(acc, f(increments[i] as number));
  }
  return acc;
}

/** The same accumulation in plain f32, for the comparison. */
export function f32Accumulate(
  start: number,
  increments: ArrayLike<number>,
): number {
  let acc = f(start);
  for (let i = 0; i < increments.length; i++) {
    acc = f(acc + f(increments[i] as number));
  }
  return acc;
}

/** And in f64, which is the answer both are trying to reach. */
export function f64Accumulate(
  start: number,
  increments: ArrayLike<number>,
): number {
  let acc = start;
  for (let i = 0; i < increments.length; i++) {
    acc += f(increments[i] as number);
  }
  return acc;
}
