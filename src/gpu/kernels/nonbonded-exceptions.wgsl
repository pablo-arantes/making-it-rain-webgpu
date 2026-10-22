// The two terms every excluded pair contributes to real space.
//
// Reciprocal space sums over *every* pair, including the ones the force field
// excludes, so each excluded pair's reciprocal contribution has to be taken
// back out here -- that is the `erf` term, computed from the particles' own
// charges. Then the exception's own interaction is applied, unscreened, with
// whatever chargeProd, sigma and epsilon the force field gave it.
//
// These are not a correction in the sense of being small. For the solvated
// protein-ligand fixture the exclusion corrections come to +1.67e6 kJ/mol,
// against a total potential energy of -1.4e5: a direct-space kernel that looked
// like a plain pair sum would be wrong by more than ten times the answer.
//
// One invocation per exception, scattering to the two atoms it touches.

@group(0) @binding(3) var<storage, read> particle: array<vec4<f32>>;      // q, sigma, eps, _
@group(0) @binding(4) var<storage, read> exceptions: array<vec4<f32>>;    // p1, p2 (bitcast), chargeProd, _
@group(0) @binding(5) var<uniform>       params: Params;

struct Params {
  exception_count: u32,
  alpha: f32,
  box: vec3<f32>,
  /// sigma and epsilon per exception live in the second half of `exceptions`.
  parameter_offset: u32,
  /// The box and half the box per axis in fixed point, for the exact minimum
  /// image; only `.xy` of each is used.
  box_fixed: array<vec4<u32>, 3>,
  half_box_fixed: array<vec4<u32>, 3>,
  /// The off-diagonal lattice components in fixed point and the inverse box in
  /// f32, appended so no existing offset moves. Zero and diagonal for a
  /// rectangular cell.
  off_fixed: array<vec4<u32>, 3>,
  inverse: array<vec4<f32>, 3>,
}

/// Whether the cell is skewed, from the off-diagonals being non-zero.
fn exception_cell_is_skewed() -> bool {
  return (params.off_fixed[0].x | params.off_fixed[0].y |
          params.off_fixed[1].x | params.off_fixed[1].y |
          params.off_fixed[2].x | params.off_fixed[2].y) != 0u;
}

/// The pair vector between two atoms, nearest image.
///
/// A skewed cell takes the general reduction, which has to see all three
/// components at once -- `delta_image_axis` returns one, which is why the
/// rectangular form can be three independent calls and this cannot.
fn exception_vector(i: u32, j: u32) -> vec3<f32> {
  if (exception_cell_is_skewed()) {
    return nearest_image_triclinic(
      array<vec2<u32>, 3>(
        sub64(positions[j * 3u + 0u], positions[i * 3u + 0u]),
        sub64(positions[j * 3u + 1u], positions[i * 3u + 1u]),
        sub64(positions[j * 3u + 2u], positions[i * 3u + 2u]),
      ),
      array<vec2<u32>, 3>(
        params.box_fixed[0].xy, params.box_fixed[1].xy, params.box_fixed[2].xy,
      ),
      array<vec2<u32>, 3>(
        params.off_fixed[0].xy, params.off_fixed[1].xy, params.off_fixed[2].xy,
      ),
      params.inverse[0].xyz, params.inverse[1].xyz, params.inverse[2].xyz,
    );
  }
  return vec3<f32>(
    delta_image_axis(i, j, 0u, params.box_fixed[0].xy, params.half_box_fixed[0].xy),
    delta_image_axis(i, j, 1u, params.box_fixed[1].xy, params.half_box_fixed[1].xy),
    delta_image_axis(i, j, 2u, params.box_fixed[2].xy, params.half_box_fixed[2].xy),
  );
}

@compute @workgroup_size(64)
fn direct_exceptions(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= params.exception_count) { return; }

  let header = exceptions[e];
  let i = bitcast<u32>(header.x);
  let j = bitcast<u32>(header.y);
  let charge_prod = header.z;
  let extra = exceptions[params.parameter_offset + e];
  let sigma = extra.x;
  let epsilon = extra.y;

  // Wrapped in fixed point before conversion, like the pair kernel: an
  // exception can span a periodic boundary, and converting a box-sized
  // difference to f32 first rounds away what the wrap would have used.
  let d = exception_vector(i, j);
  let r_squared = dot(d, d);
  let r = sqrt(r_squared);
  let alpha_r = params.alpha * r;
  var energy_sum = 0.0;
  var d_e_d_r = 0.0;

  // Take back this pair's reciprocal-space contribution, from the particles'
  // own charges rather than the exception's chargeProd.
  let particle_product = 138.93545764498457 * particle[i].x * particle[j].x;
  if (particle_product != 0.0) {
    let unscreened = erf_approx(alpha_r);
    energy_sum += -particle_product * unscreened / r;
    d_e_d_r += particle_product *
      (unscreened / r_squared - params.alpha * erf_derivative(alpha_r) / r);
  }

  // Then the exception's own interaction.
  let scaled_charge = 138.93545764498457 * charge_prod;
  if (scaled_charge != 0.0) {
    energy_sum += scaled_charge / r;
    d_e_d_r += -scaled_charge / r_squared;
  }
  if (epsilon != 0.0) {
    let s2 = sigma * sigma / r_squared;
    let s6 = s2 * s2 * s2;
    let s12 = s6 * s6;
    energy_sum += 4.0 * epsilon * (s12 - s6);
    d_e_d_r += (4.0 * epsilon / r) * (6.0 * s6 - 12.0 * s12);
  }

  accumulate_energy(energy_sum);
  let f = (d_e_d_r / r) * d;
  add_force(i, f);
  add_force(j, -f);
}
