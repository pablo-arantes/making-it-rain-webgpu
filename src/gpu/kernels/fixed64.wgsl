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

/// `v * n`, exactly, keeping the low 64 bits.
///
/// WGSL has no 64-bit multiply and no 32x32 -> 64 intrinsic, so the low word is
/// split into 16-bit halves whose four partial products each fit in 32 bits and
/// are carried by hand. Anything above 2^64 is dropped, which is what the
/// two's-complement representation wants: the result is correct modulo 2^64.
fn mul64_u32(v: vec2<u32>, n: u32) -> vec2<u32> {
  let a0 = v.x & 0xffffu;
  let a1 = v.x >> 16u;
  let b0 = n & 0xffffu;
  let b1 = n >> 16u;
  let p00 = a0 * b0;
  let p01 = a0 * b1;
  let p10 = a1 * b0;
  let p11 = a1 * b1;
  // The 2^16 column: the carry out of p00 plus the low halves of the two cross
  // terms. Three addends each below 2^17, so the sum cannot overflow.
  let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
  let low = (p00 & 0xffffu) | (mid << 16u);
  let high = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u) + v.y * n;
  return vec2<u32>(low, high);
}

/// `d` minus `whole` boxes, in the direction `quotient` points.
fn shift_whole_boxes(
  d: vec2<u32>,
  box: vec2<u32>,
  quotient: f32,
  whole: f32,
) -> vec2<u32> {
  let shift = mul64_u32(box, u32(whole));
  if (quotient < 0.0) { return add64(d, shift); }
  return sub64(d, shift);
}

/// The nearest image of a difference whose endpoints are both inside the box.
///
/// One conditional shift, and that is provably the whole minimum image: for
/// `a` and `b` in [0, box) the difference lies in (-box, box), so adding or
/// removing a single box lands it in [-half, half]. The pair kernel gets that
/// precondition from the `wrap` pass in cell-list.wgsl, which folds every
/// coordinate into [0, box) once per step.
///
/// 🔴 Never call this on raw positions. `minimum_image64` below takes any
/// input and is what the exception and restraint kernels use. This is a
/// separate function rather than a fast path inside that one because what the
/// profile measured is control flow, not arithmetic: a conditional block nested
/// in the pair kernel's innermost loop costs the backend's ability to unroll
/// and pipeline the candidate loop around it, whether or not the block runs.
///
/// The tie at exactly half a box resolves on `!less64(half, d)` here too, so
/// the two functions agree bit for bit wherever both are defined.
fn nearest_image_box(difference: vec2<u32>, box: vec2<u32>, half: vec2<u32>) -> f32 {
  var d = difference;
  if (negative64(d)) {
    if (less64(half, neg64(d))) { d = add64(d, box); }
  } else {
    if (less64(half, d)) { d = sub64(d, box); }
  }
  return from_fixed(d);
}

/// Shift a fixed-point difference into the nearest periodic image, exactly.
///
/// 🔴 The order matters, and it is the first reason this function exists. The
/// obvious way round -- convert the difference to f32, then wrap it -- rounds
/// the value while it can still be as large as the box diagonal, at the ulp of
/// a 6.5 nm number: 4.8e-7 nm. The wrap that follows cannot recover what the
/// conversion dropped. Wrapping first in integers and converting only the small
/// result keeps it. Measured on sampled pairs (test/pair-vector.test.ts):
///
///   1L2Y, 3.08 nm box   convert then wrap  rms 2.98e-8 nm   wrap then convert  5.9e-8 worst
///   3HTB, 6.58 nm box   convert then wrap  rms 8.76e-8 nm   wrap then convert  1.2e-7 worst
///
/// 🔴 Being **loop-free** is the second reason, and it is the single largest
/// performance fact measured in this engine. This used to subtract one box at a
/// time under a `for` bounded at 64, which ran zero or one times for a real
/// pair -- so it looked free, and reasoning about iteration counts said it was.
/// It was not. A nested loop in the pair kernel's innermost block stops the
/// backend unrolling and pipelining the candidate loop around it, and the cost
/// is not the iterations but everything the compiler then cannot do.
/// Measured through `web/validate.html` on an M3 Pro, 3HTB at 26,749 atoms:
///
///   loop, one box at a time          437,782 us per step in the pair kernel
///   this, loop-free and general       72,171 us                       6x
///   two unrolled shifts only          24,111 us                      18x
///   `nearest_image_box`, one shift    27,319 us                      16x
///   no wrap at all (diagnostic)        9,614 us
///
/// The hot pair kernel does not call this at all any more: it reads positions
/// already folded into the box and uses `nearest_image_box`, whose single shift
/// is provably the whole wrap. This one keeps no precondition, for the exception
/// and restraint kernels, which read unwrapped positions and together cost under
/// 100 us -- so being 3x the unrolled version costs nothing that is measured.
///
/// Skipping the estimate behind an integer `|d| < box` test was measured too and
/// bought nothing -- 72,688 us -- so there is no guard: the cost is the control
/// flow the backend cannot schedule around, not the divide.
///
/// 1L2Y barely moved through all of this, which is why profiling only the small
/// fixture missed it: 1,152 -> 1,036 us. The whole ladder, including the
/// diagnostics that attributed it, is in
/// docs/measurements/profile-step-m3pro.json.
///
/// The box count is estimated in f32 and applied in integers, so the f32 only
/// ever picks *which* multiple of the box to remove -- every operation that
/// touches the returned value is still an integer one, and the rounding this
/// function exists to avoid never reaches the result. `trunc` rather than
/// `round` leaves a residue in (-box, box) with the same sign the loop produced,
/// and the conditional shifts below finish it with the same `<=` comparison, so
/// this is bit for bit what the loop returned. f32 holds integers exactly only
/// to 2^24, so a quotient above that leaves a residue of a few boxes; the
/// estimate therefore runs twice, which covers the whole representable range.
/// Neither round runs at all for a pair inside the cutoff.
///
/// It terminates on `|d| <= half` rather than `<`, so a separation of exactly
/// half a box settles on one image instead of alternating between two equally
/// distant ones. Both give the same distance; they differ in the sign of the
/// force, and that configuration has measure zero.
fn minimum_image64(difference: vec2<u32>, box: vec2<u32>, half: vec2<u32>) -> f32 {
  var d = difference;
  let box_f32 = f32(box.y) + f32(box.x) * FIXED_SCALE_INV;

  var quotient = from_fixed(d) / box_f32;
  var whole = abs(trunc(quotient));
  if (whole != 0.0) {
    d = shift_whole_boxes(d, box, quotient, whole);
    quotient = from_fixed(d) / box_f32;
    whole = abs(trunc(quotient));
    if (whole != 0.0) { d = shift_whole_boxes(d, box, quotient, whole); }
  }

  // Two unrolled shifts: one for the residue `trunc` leaves, one in case the
  // f32 estimate was a box out either way.
  if (negative64(d)) {
    if (less64(half, neg64(d))) { d = add64(d, box); }
  } else {
    if (less64(half, d)) { d = sub64(d, box); }
  }
  if (negative64(d)) {
    if (less64(half, neg64(d))) { d = add64(d, box); }
  } else {
    if (less64(half, d)) { d = sub64(d, box); }
  }
  return from_fixed(d);
}

/// The nearest image of a fixed-point difference, for a general cell.
///
/// 🔴 Two representations of the cell, and both are necessary. The lattice
/// components arrive in **fixed point**, because the shift they produce lands
/// on a position difference that has to stay exact -- that is the whole reason
/// positions are integers. The inverse box arrives in **f32**, because it is
/// only used to *decide* which image to take.
///
/// The decision is `round(d · B⁻¹)` per fractional axis. Subtracting lattice
/// vector `c` changes only `f_c`, so the three fractional axes are independent
/// and there is no ordering and no search -- the Cartesian "reduce against c,
/// then b, then a" is exactly this. For positions folded into the cell
/// `|Δf| < 1`, so each shift is one of -1, 0, 1 and applies as an exact
/// `add64`/`sub64` with **no multiply at all**.
///
/// An f32 estimate picks that integer wrongly only within about 1.6e-7 of the
/// half-cell boundary, where the two images are equidistant anyway. The
/// comparison is strict, `> 0.5`, which is the same tie behaviour as the
/// rectangular `nearest_image_box` it generalises: at exactly half a cell the
/// reduction keeps its own image rather than alternating.
///
/// For a rectangular cell `off` is all zero, the inverse is diagonal, and this
/// reduces to one conditional shift per axis -- bit for bit what
/// `nearest_image_box` produced.
fn nearest_image_triclinic(
  d: array<vec2<u32>, 3>,
  diag: array<vec2<u32>, 3>,
  off: array<vec2<u32>, 3>,
  inverse0: vec3<f32>,
  inverse1: vec3<f32>,
  inverse2: vec3<f32>,
) -> vec3<f32> {
  var x = d[0];
  var y = d[1];
  var z = d[2];

  // f = r . M, from the f32 difference. Only the integer it implies is used.
  let rx = from_fixed(x);
  let ry = from_fixed(y);
  let rz = from_fixed(z);
  let fa = rx * inverse0.x + ry * inverse1.x + rz * inverse2.x;
  let fb = rx * inverse0.y + ry * inverse1.y + rz * inverse2.y;
  let fc = rx * inverse0.z + ry * inverse1.z + rz * inverse2.z;

  // c = (off[1], off[2], diag[2]); b = (off[0], diag[1], 0); a = (diag[0], 0, 0).
  if (fc > 0.5) {
    x = sub64(x, off[1]);
    y = sub64(y, off[2]);
    z = sub64(z, diag[2]);
  } else if (fc < -0.5) {
    x = add64(x, off[1]);
    y = add64(y, off[2]);
    z = add64(z, diag[2]);
  }
  if (fb > 0.5) {
    x = sub64(x, off[0]);
    y = sub64(y, diag[1]);
  } else if (fb < -0.5) {
    x = add64(x, off[0]);
    y = add64(y, diag[1]);
  }
  if (fa > 0.5) {
    x = sub64(x, diag[0]);
  } else if (fa < -0.5) {
    x = add64(x, diag[0]);
  }
  return vec3<f32>(from_fixed(x), from_fixed(y), from_fixed(z));
}

/// One coordinate folded into the cell, so `|Δf| < 1` holds for every pair.
///
/// Subtracts whole lattice vectors until the fractional coordinate is in
/// [0, 1). The count comes from an f32 `floor`, which only has to be close
/// because the loop checks its own work; the subtraction is exact.
fn fold_into_cell(
  r: array<vec2<u32>, 3>,
  diag: array<vec2<u32>, 3>,
  off: array<vec2<u32>, 3>,
  inverse0: vec3<f32>,
  inverse1: vec3<f32>,
  inverse2: vec3<f32>,
) -> array<vec2<u32>, 3> {
  var x = r[0];
  var y = r[1];
  var z = r[2];
  // Bounded rather than unbounded: a wild coordinate must not hang a dispatch.
  // Three axes, outermost first, because folding `c` moves `a` and `b`.
  for (var guard = 0u; guard < 64u; guard++) {
    let rx = from_fixed(x);
    let ry = from_fixed(y);
    let rz = from_fixed(z);
    let fc = rx * inverse0.z + ry * inverse1.z + rz * inverse2.z;
    if (fc >= 1.0) {
      x = sub64(x, off[1]); y = sub64(y, off[2]); z = sub64(z, diag[2]);
    } else if (fc < 0.0) {
      x = add64(x, off[1]); y = add64(y, off[2]); z = add64(z, diag[2]);
    } else { break; }
  }
  for (var guard = 0u; guard < 64u; guard++) {
    let fb = from_fixed(x) * inverse0.y + from_fixed(y) * inverse1.y
           + from_fixed(z) * inverse2.y;
    if (fb >= 1.0) {
      x = sub64(x, off[0]); y = sub64(y, diag[1]);
    } else if (fb < 0.0) {
      x = add64(x, off[0]); y = add64(y, diag[1]);
    } else { break; }
  }
  for (var guard = 0u; guard < 64u; guard++) {
    let fa = from_fixed(x) * inverse0.x + from_fixed(y) * inverse1.x
           + from_fixed(z) * inverse2.x;
    if (fa >= 1.0) { x = sub64(x, diag[0]); }
    else if (fa < 0.0) { x = add64(x, diag[0]); }
    else { break; }
  }
  return array<vec2<u32>, 3>(x, y, z);
}
