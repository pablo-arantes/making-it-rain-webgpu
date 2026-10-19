// erfc for the real-space Ewald sum.
//
// WGSL has no erfc, so this is the rational approximation from Numerical
// Recipes: a single Horner chain in t = 1/(1 + x/2), times exp(-x^2).  Its
// error is *fractional* rather than absolute, which is the property that
// matters here -- at the cutoff alpha*r reaches 2.63 where erfc is 2.0e-4, and
// an approximation good to 1e-7 absolute would be wrong there by parts in a
// thousand.
//
// `erfcApproximate` in src/reference/erf.ts is the same expression in f64, so a
// test can separate this approximation's error from f32 rounding;
// test/erf.test.ts measures both against an mpmath-pinned erfc.

fn erfc_approx(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
             t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
             t * (-0.82215223 + t * 0.17087277))))))));
  let value = t * exp(-z * z + poly);
  if (x < 0.0) { return 2.0 - value; }
  return value;
}

fn erf_approx(x: f32) -> f32 {
  return 1.0 - erfc_approx(x);
}

/// 2/sqrt(pi) exp(-x^2), the derivative of erf.
fn erf_derivative(x: f32) -> f32 {
  return 1.1283791670955126 * exp(-x * x);
}
