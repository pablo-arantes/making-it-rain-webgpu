// Position restraints: `k * periodicdistance(x, y, z, x0, y0, z0)^2`.
//
// One thread per restrained atom. The engine implements this one expression
// directly rather than through an expression parser, and `parseSystem` refuses
// a `CustomExternalForce` carrying anything else.
//
// The minimum image is taken **in fixed point**, before anything is converted
// to f32: the reference positions are stored in the same fixed point the
// coordinates are, so the difference is exact, the wrap is integer arithmetic,
// and only the small wrapped result becomes an f32. `minimum_image64` in
// fixed64.wgsl is the shared helper, and this kernel is where it was written
// first -- the nonbonded kernels used to convert before wrapping and now use
// the same function. docs/NONBONDED.md has the measurement.
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
fn restraint_delta(atom: u32, term: u32, axis: u32) -> f32 {
  return minimum_image64(
    sub64(positions[atom * 3u + axis], restraint_reference[term * 3u + axis]),
    restraint.box[axis].xy,
    restraint.half_box[axis].xy,
  );
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
