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
  /// The off-diagonal lattice components in fixed point and the inverse box in
  /// f32, appended so no existing offset moves. Zero and diagonal for a
  /// rectangular cell.
  off_box: array<vec4<u32>, 3>,
  inverse: array<vec4<f32>, 3>,
}

/// The displacement from a reference point to an atom, nearest image, in nm.
///
/// A skewed cell takes the general reduction, which needs all three components
/// together; a rectangular one keeps the per-axis `minimum_image64` it had.
///
/// 🔴 `minimum_image64` here, not `nearest_image_box`: a restraint reference is
/// an *arbitrary* point from the system the restraints were built on, not a
/// position this engine has folded into the cell, so there is no `|Δf| < 1` to
/// rely on. The general reduction handles that too -- its f32 estimate is a
/// `round`, not a one-of-three choice -- but the rectangular path needs the
/// unbounded version rather than the single shift.
fn restraint_vector(atom: u32, term: u32) -> vec3<f32> {
  let dx = sub64(positions[atom * 3u + 0u], restraint_reference[term * 3u + 0u]);
  let dy = sub64(positions[atom * 3u + 1u], restraint_reference[term * 3u + 1u]);
  let dz = sub64(positions[atom * 3u + 2u], restraint_reference[term * 3u + 2u]);
  let skewed = (restraint.off_box[0].x | restraint.off_box[0].y |
                restraint.off_box[1].x | restraint.off_box[1].y |
                restraint.off_box[2].x | restraint.off_box[2].y) != 0u;
  if (skewed) {
    return nearest_image_triclinic(
      array<vec2<u32>, 3>(dx, dy, dz),
      array<vec2<u32>, 3>(
        restraint.box[0].xy, restraint.box[1].xy, restraint.box[2].xy,
      ),
      array<vec2<u32>, 3>(
        restraint.off_box[0].xy, restraint.off_box[1].xy, restraint.off_box[2].xy,
      ),
      restraint.inverse[0].xyz, restraint.inverse[1].xyz, restraint.inverse[2].xyz,
    );
  }
  return vec3<f32>(
    minimum_image64(dx, restraint.box[0].xy, restraint.half_box[0].xy),
    minimum_image64(dy, restraint.box[1].xy, restraint.half_box[1].xy),
    minimum_image64(dz, restraint.box[2].xy, restraint.half_box[2].xy),
  );
}

@compute @workgroup_size(64)
fn position_restraint(@builtin(global_invocation_id) gid: vec3<u32>) {
  let term = gid.x;
  if (term >= restraint.count) { return; }
  let atom = restraint_terms[term].x;
  let k = bitcast<f32>(restraint_terms[term].y);

  let d = restraint_vector(atom, term);
  accumulate_energy(k * dot(d, d));
  // F = -dE/dr = -2 k d.
  add_force(atom, -2.0 * k * d);
}
