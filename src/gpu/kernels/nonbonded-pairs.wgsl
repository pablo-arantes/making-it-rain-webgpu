// Real-space Ewald and Lennard-Jones over the pairs inside the cutoff.
//
// One invocation per atom, gathering: it walks the 27 cells around its own and
// sums every interaction it finds, so the pair loop needs no atomics for its
// own force.  Every pair is therefore evaluated twice, once from each end,
// and the energy is halved to compensate.
//
// Exclusions are skipped, not zeroed. An excluded pair's reciprocal-space
// contribution is removed, and its own interaction applied, by the companion
// kernel in nonbonded-exceptions.wgsl -- together those are most of what
// OpenMM reports for this force group. See src/reference/forces/nonbonded.ts.

@group(0) @binding(3) var<storage, read> particle: array<vec4<f32>>;  // q, sigma, eps, _
@group(0) @binding(4) var<storage, read> lists: array<u32>;
@group(0) @binding(5) var<uniform>       params: Params;

struct Params {
  particle_count: u32,
  /// Where the sorted atom order begins in `lists`; the cell starts precede it.
  sorted_offset: u32,
  /// Where each atom's exclusion-list start index begins in `lists`.
  exclusion_start_offset: u32,
  /// Where the flat exclusion list begins in `lists`.
  exclusion_list_offset: u32,
  /// Where each atom's cell index begins in `lists`.
  cell_of_offset: u32,
  cells: vec3<u32>,
  cutoff: f32,
  box: vec3<f32>,
  alpha: f32,
}

/// Decompose a flat cell index back into its three axis indices.
///
/// Row-major in (x, y, z), matching `cell-list.wgsl` and the host builder.
fn cell_axes(cell: u32, cells: vec3<u32>) -> vec3<u32> {
  let z = cell % cells.z;
  let rest = cell / cells.z;
  return vec3<u32>(rest / cells.y, rest % cells.y, z);
}

/// Does `other` appear in `atom`'s exclusion list?
///
/// Checked only after the distance test, because almost every candidate fails
/// that first: a 1.0 nm sphere at water density holds about 240 atoms out of
/// the few thousand the 27 cells contain.
fn is_excluded(atom: u32, other: u32) -> bool {
  let start = lists[params.exclusion_start_offset + atom];
  let end = lists[params.exclusion_start_offset + atom + 1u];
  for (var k = start; k < end; k++) {
    if (lists[params.exclusion_list_offset + k] == other) { return true; }
  }
  return false;
}

@compute @workgroup_size(64)
fn direct_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particle_count) { return; }

  let me = particle[i];
  let cutoff_squared = params.cutoff * params.cutoff;
  let cells = params.cells;

  // Which cell this atom is in, read from the list rather than recomputed.
  //
  // 🔴 It used to be recomputed here, from the wrapped fractional coordinate in
  // f32. That is a different calculation from the one that built the list --
  // which buckets in fixed point -- and at the 6.5 nm edge of the 3HTB box an
  // f32 coordinate has a 4.8e-7 nm ulp, so the two could disagree for an atom
  // near a cell boundary. In a gather kernel that is worse than it sounds: atom
  // i would scan the neighbourhood of a cell it is not listed in, so i could
  // miss j while j still finds i. The pair contributes to one atom's force and
  // not the other's, which breaks Newton's third law and leaks momentum. Taking
  // the home cell from the list makes the two agree by construction.
  let home = cell_axes(lists[params.cell_of_offset + i], cells);

  var force_sum = vec3<f32>(0.0);
  var energy_sum = 0.0;

  for (var dx = -1; dx <= 1; dx++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dz = -1; dz <= 1; dz++) {
        let neighbour = (vec3<i32>(home) + vec3<i32>(dx, dy, dz) + vec3<i32>(cells))
                      % vec3<i32>(cells);
        let cell = (u32(neighbour.x) * cells.y + u32(neighbour.y)) * cells.z
                 + u32(neighbour.z);
        let start = lists[cell];
        let end = lists[cell + 1u];
        for (var slot = start; slot < end; slot++) {
          let j = lists[params.sorted_offset + slot];
          if (j == i) { continue; }

          let d = delta_periodic(i, j, params.box);
          let r_squared = dot(d, d);
          if (r_squared >= cutoff_squared) { continue; }
          if (is_excluded(i, j)) { continue; }

          let other = particle[j];
          let r = sqrt(r_squared);
          let alpha_r = params.alpha * r;
          let charge_product = 138.93545764498457 * me.x * other.x;
          let screened = erfc_approx(alpha_r);
          energy_sum += charge_product * screened / r;
          var d_e_d_r = charge_product *
            (-screened / r_squared - params.alpha * erf_derivative(alpha_r) / r);

          let epsilon = sqrt(me.z * other.z);
          if (epsilon != 0.0) {
            let sigma = 0.5 * (me.y + other.y);
            let s2 = sigma * sigma / r_squared;
            let s6 = s2 * s2 * s2;
            let s12 = s6 * s6;
            energy_sum += 4.0 * epsilon * (s12 - s6);
            d_e_d_r += (4.0 * epsilon / r) * (6.0 * s6 - 12.0 * s12);
          }
          // F_i = (dE/dr / r) * d, and this invocation owns only atom i.
          force_sum += (d_e_d_r / r) * d;
        }
      }
    }
  }

  add_force(i, force_sum);
  // Each pair was seen from both ends, so the energy is counted twice.
  accumulate_energy(0.5 * energy_sum);
}
