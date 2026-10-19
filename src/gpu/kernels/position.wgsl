// Position integration in 64-bit fixed point, and the diagnostic for the
// floating-point scheme it replaced.
//
// Positions cannot be plain f32.  The notebook sets
// `integrator.setConstraintTolerance(1e-6)`, which on a 0.1 nm X-H bond is
// 1e-7 nm, while an f32 holding a 6.5 nm coordinate -- the 3HTB box is
// 6.52 nm -- has an ulp of 4.77e-7 nm.  An increment that small does not drift
// an f32 coordinate, it leaves it completely unchanged.
//
// They are not double-single either, which was the first plan: see
// docs/PRECISION.md and the diagnostic at the bottom of this file.  They are
// the same fixed-point integers the forces use, with one addition the forces
// do not need: the integrator carries the part of each displacement that did
// not fit the format into the next step, which bounds the drift of a long run
// instead of letting it grow with the step count.

@group(0) @binding(0) var<storage, read>       increments: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@group(0) @binding(2) var<uniform>             params: Params;

struct Params {
  increment_count: u32,
  // The starting coordinate.  The error of any scheme depends on the magnitude
  // the increments are added to, so it is an input rather than a constant.
  start: f32,
}

// out[0], out[1]   plain fixed-point position, low and high words
// out[2], out[3]   residual-compensated fixed-point position, low and high
// out[4]           the same sum accumulated in plain f32, as a bit pattern
// out[5], out[6]   the Dekker double-single accumulator, hi and lo, as bits
// out[7]           the residual still outstanding at the end, as a bit pattern
@compute @workgroup_size(1)
fn main() {
  var plain = to_fixed(params.start);
  var compensated = to_fixed(params.start);
  var residual = 0.0;
  var single = params.start;
  var ds = vec2<f32>(params.start, 0.0);

  for (var i = 0u; i < params.increment_count; i++) {
    let dx = increments[i];

    // Plain: convert each displacement and add it.  The conversion error is at
    // most half the format's resolution, but for a repeated displacement it is
    // the *same* error every step, so it accumulates linearly in the step
    // count instead of random-walking.
    plain = add64(plain, to_fixed(dx));

    // Compensated: carry what did not fit into the next step.  The outstanding
    // residual is bounded by one unit in the last place forever, so the drift
    // stops growing with the step count.  `from_fixed` is an integer-to-float
    // conversion, so no optimiser can fold this subtraction the way it folds
    // Dekker's (see the diagnostic below).
    let wanted = dx + residual;
    let taken = to_fixed(wanted);
    compensated = add64(compensated, taken);
    residual = wanted - from_fixed(taken);

    single = single + dx;
    ds = ds_add_f32(ds, dx);
  }

  out[0] = plain.x;
  out[1] = plain.y;
  out[2] = compensated.x;
  out[3] = compensated.y;
  out[4] = bitcast<u32>(single);
  out[5] = bitcast<u32>(ds.x);
  out[6] = bitcast<u32>(ds.y);
  out[7] = bitcast<u32>(residual);
}

// ---------------------------------------------------------------------------
// Dekker error-free arithmetic, kept as a *diagnostic* and used nowhere else.
//
// `two_sum` returns a sum and the bits that sum dropped, which is what makes a
// double-single representation possible -- on hardware that computes it. On
// Chrome/Metal, with runtime inputs, the error term comes back as exactly
// zero: the backend simplifies `b - (s - a)` to `b - b`, which is true in
// exact arithmetic and destroys the one thing the expression exists to
// measure. The check in src/gpu/checks.ts reports what each device does, so if
// a backend ever computes it correctly we find out, and so the reason
// positions are integers stays measured rather than remembered.
// ---------------------------------------------------------------------------

fn two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = a + b;
  let bb = s - a;
  let err = (a - (s - bb)) + (b - bb);
  return vec2<f32>(s, err);
}

fn quick_two_sum(a: f32, b: f32) -> vec2<f32> {
  let s = a + b;
  return vec2<f32>(s, b - (s - a));
}

fn ds_add_f32(a: vec2<f32>, b: f32) -> vec2<f32> {
  let s = two_sum(a.x, b);
  return quick_two_sum(s.x, s.y + a.y);
}
