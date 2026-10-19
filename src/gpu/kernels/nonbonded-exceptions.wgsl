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

  let d = delta_periodic(i, j, params.box);
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
