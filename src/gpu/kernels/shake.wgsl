// SHAKE: the iterative constraint solver, one thread per cluster.
//
// Constraints in different clusters never interact, so a thread can load its
// whole cluster into registers and iterate it to convergence with no atomics
// and no cross-thread coupling. Measured on both fixtures, the largest cluster
// is 4 atoms and 3 constraints -- `splitConstraints` refuses anything above
// the capacity below rather than truncating.
//
// The deltas within a cluster are computed from the same trial positions and
// only then applied, which is the Jacobi ordering
// `ReferenceCCMAAlgorithm::applyConstraints` uses. CCMA's own refinement, a
// premultiply by an approximate inverse of the coupling matrix, is not here:
// it changes the rate of convergence, not the fixed point, and these clusters
// have very little coupling to accelerate.
//
// Positions are held as displacements from the reference, in f32. A trial bond
// vector is then `reference_delta + (d_i - d_j)`, both small, so the absolute
// coordinates never leave fixed point.

@group(0) @binding(0) var<storage, read>       reference_position: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> trial: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>       inverse_mass: array<f32>;
@group(0) @binding(3) var<storage, read>       headers: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       cluster_atoms: array<u32>;
@group(0) @binding(5) var<storage, read>       cluster_constraints: array<vec4<f32>>;
@group(0) @binding(6) var<uniform>             params: Params;

struct Params {
  cluster_count: u32,
  max_iterations: u32,
  /// Relative, as OpenMM's is: a fraction of the constrained distance.
  tolerance: f32,
  /// 0 constrains positions, 1 constrains velocities.
  velocity_mode: u32,
}

const MAX_ATOMS: u32 = 8u;
const MAX_CONSTRAINTS: u32 = 8u;

@compute @workgroup_size(64)
fn shake_cluster(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.cluster_count) { return; }
  let header = headers[index];
  let atom_count = header.x;
  let constraint_count = header.y;
  let atom_offset = header.z;
  let constraint_offset = header.w;

  // Load the cluster.
  var global_atom: array<u32, 8>;
  var state: array<vec3<f32>, 8>;
  var inverse: array<f32, 8>;
  for (var a = 0u; a < MAX_ATOMS; a++) {
    if (a >= atom_count) { break; }
    let atom = cluster_atoms[atom_offset + a];
    global_atom[a] = atom;
    inverse[a] = inverse_mass[atom];
    // Positions: the displacement from the reference. Velocities: the value
    // itself, which `trial` carries in that mode.
    if (params.velocity_mode == 1u) {
      state[a] = vec3<f32>(
        from_fixed(trial[atom * 3u + 0u]),
        from_fixed(trial[atom * 3u + 1u]),
        from_fixed(trial[atom * 3u + 2u]),
      );
    } else {
      state[a] = vec3<f32>(
        from_fixed(sub64(trial[atom * 3u + 0u], reference_position[atom * 3u + 0u])),
        from_fixed(sub64(trial[atom * 3u + 1u], reference_position[atom * 3u + 1u])),
        from_fixed(sub64(trial[atom * 3u + 2u], reference_position[atom * 3u + 2u])),
      );
    }
  }

  // Per-constraint fixed quantities: the local atom pair, the target distance distance,
  // the reference bond vector and the reduced mass.
  var local_i: array<u32, 8>;
  var local_j: array<u32, 8>;
  var target_distance: array<f32, 8>;
  var reference_vector: array<vec3<f32>, 8>;
  var reduced: array<f32, 8>;
  for (var c = 0u; c < MAX_CONSTRAINTS; c++) {
    if (c >= constraint_count) { break; }
    let packed = cluster_constraints[constraint_offset + c];
    let li = bitcast<u32>(packed.x);
    let lj = bitcast<u32>(packed.y);
    local_i[c] = li;
    local_j[c] = lj;
    target_distance[c] = packed.z;
    let gi = global_atom[li];
    let gj = global_atom[lj];
    reference_vector[c] = vec3<f32>(
      from_fixed(sub64(reference_position[gi * 3u + 0u], reference_position[gj * 3u + 0u])),
      from_fixed(sub64(reference_position[gi * 3u + 1u], reference_position[gj * 3u + 1u])),
      from_fixed(sub64(reference_position[gi * 3u + 2u], reference_position[gj * 3u + 2u])),
    );
    let sum = inverse[li] + inverse[lj];
    if (sum == 0.0) { reduced[c] = 0.0; } else { reduced[c] = 0.5 / sum; }
  }

  let lower = 1.0 - 2.0 * params.tolerance + params.tolerance * params.tolerance;
  let upper = 1.0 + 2.0 * params.tolerance + params.tolerance * params.tolerance;

  for (var iteration = 0u; iteration < params.max_iterations; iteration++) {
    var delta: array<f32, 8>;
    var converged = 0u;
    for (var c = 0u; c < MAX_CONSTRAINTS; c++) {
      if (c >= constraint_count) { break; }
      let li = local_i[c];
      let lj = local_j[c];
      let r = reference_vector[c];
      if (params.velocity_mode == 1u) {
        // The relative velocity along the bond, which must go to zero. Divided
        // by the *reference* squared length, not the target distance distance squared.
        let projection = dot(state[li] - state[lj], r);
        let lengthSquared = dot(r, r);
        var d = 0.0;
        if (lengthSquared != 0.0) { d = -2.0 * reduced[c] * projection / lengthSquared; }
        delta[c] = d;
        if (abs(d) <= params.tolerance) { converged = converged + 1u; }
      } else {
        // The trial bond vector, built from small pieces.
        let rp = r + (state[li] - state[lj]);
        let trialSquared = dot(rp, rp);
        let targetSquared = target_distance[c] * target_distance[c];
        let projection = dot(rp, r);
        var d = 0.0;
        if (projection != 0.0) {
          d = reduced[c] * (targetSquared - trialSquared) / projection;
        }
        delta[c] = d;
        if (trialSquared >= lower * targetSquared && trialSquared <= upper * targetSquared) {
          converged = converged + 1u;
        }
      }
    }
    if (converged == constraint_count) { break; }
    // All deltas computed, now all applied.
    for (var c = 0u; c < MAX_CONSTRAINTS; c++) {
      if (c >= constraint_count) { break; }
      let li = local_i[c];
      let lj = local_j[c];
      let dr = reference_vector[c] * delta[c];
      state[li] = state[li] + dr * inverse[li];
      state[lj] = state[lj] - dr * inverse[lj];
    }
  }

  // Write back.
  for (var a = 0u; a < MAX_ATOMS; a++) {
    if (a >= atom_count) { break; }
    let atom = global_atom[a];
    let value = state[a];
    for (var axis = 0u; axis < 3u; axis++) {
      var component = value.x;
      if (axis == 1u) { component = value.y; }
      if (axis == 2u) { component = value.z; }
      let slot = atom * 3u + axis;
      if (params.velocity_mode == 1u) {
        trial[slot] = to_fixed(component);
      } else {
        trial[slot] = add64(reference_position[slot], to_fixed(component));
      }
    }
  }
}
