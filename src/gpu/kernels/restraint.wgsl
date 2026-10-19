// Position restraints: `k * periodicdistance(x, y, z, x0, y0, z0)^2`.
//
// One thread per restrained atom. The engine implements this one expression
// directly rather than through an expression parser, and `parseSystem` refuses
// a `CustomExternalForce` carrying anything else.
//
// The minimum image is taken **in fixed point**, before anything is converted
// to f32, which is what makes this kernel differ from `delta_periodic` in
// accumulate.wgsl. That one converts a pair difference while it can still be as
// large as the box diagonal, rounding it at the ulp of a 6.5 nm number --
// 4.8e-7 nm -- and the wrap that follows cannot recover what the conversion
// dropped (docs/NONBONDED.md). Here the reference positions are stored in the
// same fixed point the coordinates are, so the difference is exact, the wrap is
// integer arithmetic, and only the small wrapped result is converted. It is also
// a worked example of the fix `delta_periodic` still needs.
//
// 🔴 The wrap is the part of this force that can be wrong invisibly. Without it
// every atom that has not crossed a boundary gives an identical energy, so a
// freshly minimised structure looks perfect until something drifts across and
// gets pulled by `k` times a box length. The fixture's "wrapped" case displaces
// atoms by a whole box for exactly that reason.

@group(0) @binding(3) var<storage, read> restraint_terms: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> restraint_reference: array<vec2<u32>>;
@group(0) @binding(5) var<uniform>       restraint: RestraintParams;

struct RestraintParams {
  count: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  // The box per axis in fixed point, and half of it, as `vec4` because WGSL
  // gives a uniform array element a 16-byte stride and writing the padding
  // into the type keeps the host's packing and the shader's view together.
  // Only `.xy` is used: low word then high.
  box: array<vec4<u32>, 3>,
  half_box: array<vec4<u32>, 3>,
}

/// The displacement from a reference point to an atom, nearest image, in nm.
///
/// Shifted by whole boxes until the magnitude is at most half a box. The loop
/// normally runs zero times -- a restrained atom sits a fraction of a nanometre
/// from where it started -- and is bounded so that a wild coordinate cannot
/// hang the dispatch.
fn restraint_delta(atom: u32, term: u32, axis: u32) -> f32 {
  let box = restraint.box[axis].xy;
  let half = restraint.half_box[axis].xy;
  var d = sub64(positions[atom * 3u + axis], restraint_reference[term * 3u + axis]);
  for (var guard = 0u; guard < 64u; guard++) {
    if (negative64(d)) {
      // Terminate on `|d| <= half` rather than `<`, so a displacement of
      // exactly half a box settles instead of alternating between the two
      // images of equal distance. The energy is the same either way; the sign
      // of the force is not, and that configuration has measure zero.
      if (!less64(half, neg64(d))) { break; }
      d = add64(d, box);
    } else {
      if (!less64(half, d)) { break; }
      d = sub64(d, box);
    }
  }
  return from_fixed(d);
}

@compute @workgroup_size(64)
fn position_restraint(@builtin(global_invocation_id) gid: vec3<u32>) {
  let term = gid.x;
  if (term >= restraint.count) { return; }
  let atom = restraint_terms[term].x;
  let k = bitcast<f32>(restraint_terms[term].y);

  let d = vec3<f32>(
    restraint_delta(atom, term, 0u),
    restraint_delta(atom, term, 1u),
    restraint_delta(atom, term, 2u),
  );
  accumulate_energy(k * dot(d, d));
  // F = -dE/dr = -2 k d.
  add_force(atom, -2.0 * k * d);
}
