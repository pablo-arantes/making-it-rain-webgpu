// Shared 64-bit fixed-point arithmetic, held as two 32-bit words.
//
// WGSL has no include directive, so this file is prepended to every kernel
// that needs it by `composeKernel` in src/gpu/kernels.ts.  It is the one
// canonical copy; nothing duplicates it.
//
//     value = (f64(i32 high) * 2^32 + f64(u32 low)) / 2^32
//
// Range +/-2^31, resolution 2^-32 (about 2.3e-10).  Both the forces
// (kJ/mol/nm) and the positions (nm) use this format, for the same reason: it
// is integer arithmetic, so it is exact, order-independent, and no compiler
// can algebraically simplify it.
//
// That last property is not theoretical.  See docs/PRECISION.md: on
// Chrome/Metal the error term of Dekker's `two_sum` comes back as exactly zero
// for runtime inputs, because the backend folds `b - (s - a)` to `b - b`.  Any
// scheme that depends on the rounding of an f32 operation surviving the
// optimiser is unsafe here; integers are not.

const FIXED_SCALE: f32 = 4294967296.0;      // 2^32
const FIXED_SCALE_INV: f32 = 2.3283064365386963e-10;  // 2^-32

/// Negate a 64-bit value: invert both words and add one.  The carry out of the
/// low word happens exactly when `~low + 1` wraps, which is when `low` is zero.
fn neg64(v: vec2<u32>) -> vec2<u32> {
  var high = ~v.y;
  if (v.x == 0u) { high = high + 1u; }
  return vec2<u32>(~v.x + 1u, high);
}

/// Add two 64-bit values, carrying by hand.
fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  let low = a.x + b.x;
  var high = a.y + b.y;
  if (low < a.x) { high = high + 1u; }      // unsigned wrap means carry
  return vec2<u32>(low, high);
}

fn sub64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  return add64(a, neg64(b));
}

/// Split an f32 into the two words of its fixed-point representation.
///
/// Two deliberate choices, both measured.
///
/// The magnitude is decomposed and the result negated, rather than flooring a
/// signed value.  Flooring is not symmetric in f32: for a small negative x,
/// `x - floor(x)` is `1 + x`, and an f32 near 1 cannot keep the low bits of x.
/// Measured at 2.2e-8 error against 2.3e-10 for the same magnitude positive.
///
/// The fractional word rounds to nearest rather than truncating.  Truncation
/// is biased -- always toward zero -- and a bias does not average out: 200000
/// displacements of 1e-7 nm, which is 400 ps of one coordinate at a 2 fs
/// timestep, drifted 2.3e-5 nm low, and that grows linearly with simulation
/// length.  Rounding to nearest leaves a +/-2^-33 error with no preferred
/// direction, which random-walks instead.
///
/// `round` resolves a tie to the even value, and
/// src/reference/fixed-point.ts does the same, so the two agree bit for bit.
/// The product cannot reach 2^32: the largest f32 below 1 is 1 - 2^-24, so
/// `frac * 2^32` is at most 2^32 - 256 and the low word never overflows.
fn to_fixed(x: f32) -> vec2<u32> {
  let magnitude = abs(x);
  let whole = floor(magnitude);
  let frac = magnitude - whole;             // exact: both operands non-negative
  let words = vec2<u32>(u32(round(frac * FIXED_SCALE)), u32(i32(whole)));
  if (x < 0.0) { return neg64(words); }
  return words;
}

/// Collapse a 64-bit fixed-point value to f32.
///
/// Only for a value already small enough that f32 can hold it -- a force, or a
/// *difference* of positions, which is at most the cutoff.  Never call this on
/// an absolute position: that is the whole reason positions are stored this
/// way.  Take the difference in fixed point first, then convert.
///
/// The magnitude is converted and the sign reapplied, for the same reason
/// `to_fixed` decomposes a magnitude.  A negative value's two's-complement
/// form puts a large number in the low word -- -1e-9 is high = -1, low =
/// 2^32 - 4 -- and `f32(low)` cannot hold 32 bits.  Converting the sign-magnitude
/// form instead keeps the error symmetric, which matters because this feeds the
/// integrator's residual: an asymmetric conversion makes the residual sign
/// dependent, and that is a bias, not noise.
fn from_fixed(v: vec2<u32>) -> f32 {
  let negative = (v.y & 0x80000000u) != 0u;
  var m = v;
  if (negative) { m = neg64(v); }
  let value = f32(m.y) + f32(m.x) * FIXED_SCALE_INV;
  if (negative) { return -value; }
  return value;
}

/// Whether a fixed-point value is negative: the sign bit of the high word.
fn negative64(v: vec2<u32>) -> bool {
  return (v.y & 0x80000000u) != 0u;
}

/// Unsigned 64-bit `a < b`, for values already known to be non-negative.
///
/// Used by the cell list, which wraps a coordinate into [0, box) and then
/// buckets it by comparing against accumulated multiples of the cell size.
/// Doing that in integers rather than in f32 matters: an absolute coordinate
/// converted to f32 has a 4.8e-7 nm ulp at the 6.5 nm edge of the 3HTB box,
/// which is enough to put an atom in the wrong cell and lose a pair just inside
/// the cutoff. Comparisons are exact, so the only error left is the 2.3e-10 nm
/// the position already carries.
fn less64(a: vec2<u32>, b: vec2<u32>) -> bool {
  if (a.y != b.y) { return a.y < b.y; }
  return a.x < b.x;
}

/// Shift a fixed-point difference into the nearest periodic image, exactly.
///
/// 🔴 The order matters, and it is the whole reason this function exists. The
/// obvious way round -- convert the difference to f32, then wrap it -- rounds
/// the value while it can still be as large as the box diagonal, at the ulp of
/// a 6.5 nm number: 4.8e-7 nm. The wrap that follows cannot recover what the
/// conversion dropped. Wrapping first in integers and converting only the small
/// result keeps it. Measured on sampled pairs (test/pair-vector.test.ts):
///
///   1L2Y, 3.08 nm box   convert then wrap  rms 2.98e-8 nm   wrap then convert  5.9e-8 worst
///   3HTB, 6.58 nm box   convert then wrap  rms 8.76e-8 nm   wrap then convert  1.2e-7 worst
///
/// The loop runs zero times for a pair inside the cutoff and once for a pair
/// across a boundary, and is bounded so that a wild coordinate cannot hang a
/// dispatch. `half` is passed in rather than derived because a 64-bit shift is
/// more code than a second uniform field.
///
/// It terminates on `|d| <= half` rather than `<`, so a separation of exactly
/// half a box settles on one image instead of alternating between two equally
/// distant ones. Both give the same distance; they differ in the sign of the
/// force, and that configuration has measure zero.
fn minimum_image64(difference: vec2<u32>, box: vec2<u32>, half: vec2<u32>) -> f32 {
  var d = difference;
  for (var guard = 0u; guard < 64u; guard++) {
    if (negative64(d)) {
      if (!less64(half, neg64(d))) { break; }
      d = add64(d, box);
    } else {
      if (!less64(half, d)) { break; }
      d = sub64(d, box);
    }
  }
  return from_fixed(d);
}
