/**
 * erf and erfc in double precision.
 *
 * JavaScript has no `Math.erf`, and the real-space Ewald sum is built on erfc,
 * so the reference needs its own.  This is the incomplete gamma function route:
 * for x >= 0, `erfc(x) = Q(1/2, x^2)`, computed by its series below the
 * crossover and by a continued fraction above it.  Accurate to near machine
 * precision, which matters because this is the yardstick the f32 kernel
 * approximation is measured against.
 *
 * The WGSL kernel cannot use this -- no loops of unbounded length, no f64 --
 * and uses a rational approximation instead; `test/erf.test.ts` measures the
 * difference between them.
 */

/** 1/sqrt(pi) */
const INV_SQRT_PI = 0.5641895835477563;
/** ln(sqrt(pi)) */
const LN_SQRT_PI = 0.5723649429247001;
const MAX_ITERATIONS = 300;
const EPSILON = Number.EPSILON;
/** The smallest representable positive normal, for the continued fraction. */
const TINY = 1e-300;

/**
 * The regularised lower incomplete gamma P(a, x), by its series.
 * Used for x < a + 1, where the series converges fastest.
 */
function lowerGammaSeries(a: number, x: number): number {
  let term = 1 / a;
  let sum = term;
  let ap = a;
  for (let i = 0; i < MAX_ITERATIONS; i++) {
    ap += 1;
    term *= x / ap;
    sum += term;
    if (Math.abs(term) < Math.abs(sum) * EPSILON) break;
  }
  // exp(-x + a ln x - ln Gamma(a)), with ln Gamma(1/2) = ln sqrt(pi).
  return sum * Math.exp(-x + a * Math.log(x) - LN_SQRT_PI);
}

/**
 * The regularised upper incomplete gamma Q(a, x), by the modified Lentz
 * continued fraction.  Used for x >= a + 1.
 */
function upperGammaContinuedFraction(a: number, x: number): number {
  let b = x + 1 - a;
  let c = 1 / TINY;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i <= MAX_ITERATIONS; i++) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) <= EPSILON) break;
  }
  return Math.exp(-x + a * Math.log(x) - LN_SQRT_PI) * h;
}

/** The complementary error function, 1 - erf(x). */
export function erfc(x: number): number {
  if (x === 0) return 1;
  if (!Number.isFinite(x)) return x > 0 ? 0 : 2;
  const magnitude = Math.abs(x);
  const xSquared = magnitude * magnitude;
  // a = 1/2, so the crossover x^2 < a + 1 is x^2 < 1.5.
  const value =
    xSquared < 1.5
      ? 1 - lowerGammaSeries(0.5, xSquared)
      : upperGammaContinuedFraction(0.5, xSquared);
  return x >= 0 ? value : 2 - value;
}

/** The error function. */
export function erf(x: number): number {
  if (x === 0) return 0;
  const magnitude = Math.abs(x);
  const xSquared = magnitude * magnitude;
  const value =
    xSquared < 1.5
      ? lowerGammaSeries(0.5, xSquared)
      : 1 - upperGammaContinuedFraction(0.5, xSquared);
  return x >= 0 ? value : -value;
}

/**
 * The derivative of erf, 2/sqrt(pi) exp(-x^2).
 *
 * The real-space Ewald force needs it: d/dr [erfc(alpha r)/r] carries a term
 * -2 alpha/sqrt(pi) exp(-alpha^2 r^2) / r.
 */
export const erfDerivative = (x: number): number =>
  2 * INV_SQRT_PI * Math.exp(-x * x);

/**
 * The approximation the WGSL kernel uses, in f64, so a test can separate the
 * approximation's error from f32 rounding.
 *
 * A rational form in t = 2/(2 + x) whose fractional error stays below about
 * 1.2e-7 across the whole range -- relative, not absolute, which is what
 * matters here: at the cutoff alpha*r reaches 2.6 where erfc is 2e-4, and an
 * approximation with 1e-7 *absolute* error would be wrong there by parts in a
 * thousand.
 */
export function erfcApproximate(x: number): number {
  const z = Math.abs(x);
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;
  const coefficients = [
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
    4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8, 6.529054439e-9,
    5.059343495e-9, -9.91364156e-10, -2.27365122e-10, 9.6467911e-11,
    2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13, -1.12708e-13,
    3.81e-16, 7.106e-15,
  ];
  let d = 0;
  let dd = 0;
  for (let j = coefficients.length - 1; j > 0; j--) {
    const previous = d;
    d = ty * d - dd + (coefficients[j] as number);
    dd = previous;
  }
  const value = t * Math.exp(-z * z + 0.5 * ((coefficients[0] as number) + ty * d) - dd);
  return x >= 0 ? value : 2 - value;
}
