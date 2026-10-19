// SHAKE and RATTLE: the iterative constraint solvers, one thread per cluster.
//
// Constraints in different clusters never interact, so a thread can load its
// whole cluster into registers and iterate it to convergence with no atomics
// and no cross-thread coupling. Measured on both fixtures, the largest cluster
// is 4 atoms and 3 constraints -- `splitConstraints` refuses anything above
// the capacity below rather than truncating.
//
// The deltas within a cluster are computed from the same trial state and only
// then applied, which is the Jacobi ordering
// `ReferenceCCMAAlgorithm::applyConstraints` uses. CCMA's own refinement, a
// premultiply by an approximate inverse of the coupling matrix, is not here:
// it changes the rate of convergence, not the fixed point, and these clusters
// have very little coupling to accelerate.
//
// Two entry points rather than one with a mode flag, because the two solvers
// take their state from different buffers. A position is fixed point and is
// carried here as an f32 *displacement from the reference*, so that a trial
// bond vector is `reference_delta + (d_i - d_j)` -- both small -- and the
// absolute coordinates never leave fixed point. A velocity is f32 on the
// device to begin with (src/gpu/kernels/integrate.wgsl says why), and is used
// as it stands. One kernel reading one binding as both would be reading an f32
// array as fixed point, which is how this was first written: it gave NaN
// velocities and positions 2^31 out, the fixed-point range, in the browser
// lane. Separate bindings make the mistake unrepresentable.

@group(0) @binding(0) var<storage, read>       reference_position: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> trial: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read>       inverse_mass: array<f32>;
@group(0) @binding(3) var<storage, read>       headers: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read>       cluster_atoms: array<u32>;
@group(0) @binding(5) var<storage, read>       cluster_constraints: array<vec4<f32>>;
@group(0) @binding(6) var<uniform>             params: Params;
@group(0) @binding(7) var<storage, read_write> velocities: array<f32>;

struct Params {
  cluster_count: u32,
  max_iterations: u32,
  /// Relative, as OpenMM's is: a fraction of the constrained distance.
  tolerance: f32,
  pad: u32,
}

const MAX_ATOMS: u32 = 8u;
const MAX_CONSTRAINTS: u32 = 8u;

/// A cluster's constraint table, decoded once per thread.
struct Cluster {
  atom_count: u32,
  constraint_count: u32,
  global_atom: array<u32, 8>,
  inverse: array<f32, 8>,
  local_i: array<u32, 8>,
  local_j: array<u32, 8>,
  target_distance: array<f32, 8>,
  reference_vector: array<vec3<f32>, 8>,
  reduced: array<f32, 8>,
}

/// Everything that depends on the system and the reference positions, but not
/// on whatever is being projected.
fn load_cluster(index: u32) -> Cluster {
  var cluster: Cluster;
  let header = headers[index];
  cluster.atom_count = header.x;
  cluster.constraint_count = header.y;
  let atom_offset = header.z;
  let constraint_offset = header.w;

  for (var a = 0u; a < MAX_ATOMS; a++) {
    if (a >= cluster.atom_count) { break; }
    let atom = cluster_atoms[atom_offset + a];
    cluster.global_atom[a] = atom;
    cluster.inverse[a] = inverse_mass[atom];
  }

  for (var c = 0u; c < MAX_CONSTRAINTS; c++) {
    if (c >= cluster.constraint_count) { break; }
    let packed = cluster_constraints[constraint_offset + c];
    // Columns 0 and 1 are local atom indices travelling as raw bits.
    let li = bitcast<u32>(packed.x);
    let lj = bitcast<u32>(packed.y);
    cluster.local_i[c] = li;
    cluster.local_j[c] = lj;
    cluster.target_distance[c] = packed.z;
    let gi = cluster.global_atom[li];
    let gj = cluster.global_atom[lj];
    cluster.reference_vector[c] = vec3<f32>(
      from_fixed(sub64(reference_position[gi * 3u + 0u], reference_position[gj * 3u + 0u])),
      from_fixed(sub64(reference_position[gi * 3u + 1u], reference_position[gj * 3u + 1u])),
      from_fixed(sub64(reference_position[gi * 3u + 2u], reference_position[gj * 3u + 2u])),
    );
    let sum = cluster.inverse[li] + cluster.inverse[lj];
    if (sum == 0.0) { cluster.reduced[c] = 0.0; } else { cluster.reduced[c] = 0.5 / sum; }
  }
  return cluster;
}

/// Spread one sweep's deltas over the cluster's atoms, along the reference
/// bond vectors. Shared by both solvers: only the deltas differ.
fn apply_deltas(
  cluster: ptr<function, Cluster>,
  state: ptr<function, array<vec3<f32>, 8>>,
  delta: ptr<function, array<f32, 8>>,
) {
  for (var c = 0u; c < MAX_CONSTRAINTS; c++) {
    if (c >= (*cluster).constraint_count) { break; }
    let li = (*cluster).local_i[c];
    let lj = (*cluster).local_j[c];
    let dr = (*cluster).reference_vector[c] * (*delta)[c];
    (*state)[li] = (*state)[li] + dr * (*cluster).inverse[li];
    (*state)[lj] = (*state)[lj] - dr * (*cluster).inverse[lj];
  }
}

/// Pull one component out of a vec3 by a runtime index.
fn component(value: vec3<f32>, axis: u32) -> f32 {
  if (axis == 0u) { return value.x; }
  if (axis == 1u) { return value.y; }
  return value.z;
}

/// SHAKE: project the trial positions onto the constraint surface.
@compute @workgroup_size(64)
fn shake_cluster(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.cluster_count) { return; }
  var cluster = load_cluster(index);

  var state: array<vec3<f32>, 8>;
  for (var a = 0u; a < MAX_ATOMS; a++) {
    if (a >= cluster.atom_count) { break; }
    let atom = cluster.global_atom[a];
    state[a] = vec3<f32>(
      from_fixed(sub64(trial[atom * 3u + 0u], reference_position[atom * 3u + 0u])),
      from_fixed(sub64(trial[atom * 3u + 1u], reference_position[atom * 3u + 1u])),
      from_fixed(sub64(trial[atom * 3u + 2u], reference_position[atom * 3u + 2u])),
    );
  }

  // OpenMM's convergence window, on the squared length, so no square root is
  // needed in the inner loop.
  let lower = 1.0 - 2.0 * params.tolerance + params.tolerance * params.tolerance;
  let upper = 1.0 + 2.0 * params.tolerance + params.tolerance * params.tolerance;

  for (var iteration = 0u; iteration < params.max_iterations; iteration++) {
    var delta: array<f32, 8>;
    var converged = 0u;
    for (var c = 0u; c < MAX_CONSTRAINTS; c++) {
      if (c >= cluster.constraint_count) { break; }
      let r = cluster.reference_vector[c];
      // The trial bond vector, built from small pieces.
      let rp = r + (state[cluster.local_i[c]] - state[cluster.local_j[c]]);
      let trial_squared = dot(rp, rp);
      let target_squared = cluster.target_distance[c] * cluster.target_distance[c];
      let projection = dot(rp, r);
      var d = 0.0;
      if (projection != 0.0) {
        d = cluster.reduced[c] * (target_squared - trial_squared) / projection;
      }
      delta[c] = d;
      if (trial_squared >= lower * target_squared && trial_squared <= upper * target_squared) {
        converged = converged + 1u;
      }
    }
    if (converged == cluster.constraint_count) { break; }
    apply_deltas(&cluster, &state, &delta);
  }

  for (var a = 0u; a < MAX_ATOMS; a++) {
    if (a >= cluster.atom_count) { break; }
    let atom = cluster.global_atom[a];
    for (var axis = 0u; axis < 3u; axis++) {
      let slot = atom * 3u + axis;
      trial[slot] = add64(reference_position[slot], to_fixed(component(state[a], axis)));
    }
  }
}

/// RATTLE: remove the component of each constrained pair's relative velocity
/// along its bond.
///
/// Note the division by the *reference* squared length rather than by the
/// target distance squared -- that is what `applyVelocityConstraints` does, and
/// the two differ by however far the positions sit from the constraint surface.
@compute @workgroup_size(64)
fn rattle_cluster(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.cluster_count) { return; }
  var cluster = load_cluster(index);

  var state: array<vec3<f32>, 8>;
  for (var a = 0u; a < MAX_ATOMS; a++) {
    if (a >= cluster.atom_count) { break; }
    let atom = cluster.global_atom[a];
    state[a] = vec3<f32>(
      velocities[atom * 3u + 0u],
      velocities[atom * 3u + 1u],
      velocities[atom * 3u + 2u],
    );
  }

  for (var iteration = 0u; iteration < params.max_iterations; iteration++) {
    var delta: array<f32, 8>;
    var converged = 0u;
    for (var c = 0u; c < MAX_CONSTRAINTS; c++) {
      if (c >= cluster.constraint_count) { break; }
      let r = cluster.reference_vector[c];
      let projection = dot(state[cluster.local_i[c]] - state[cluster.local_j[c]], r);
      let length_squared = dot(r, r);
      var d = 0.0;
      if (length_squared != 0.0) {
        d = -2.0 * cluster.reduced[c] * projection / length_squared;
      }
      delta[c] = d;
      if (abs(d) <= params.tolerance) { converged = converged + 1u; }
    }
    if (converged == cluster.constraint_count) { break; }
    apply_deltas(&cluster, &state, &delta);
  }

  for (var a = 0u; a < MAX_ATOMS; a++) {
    if (a >= cluster.atom_count) { break; }
    let atom = cluster.global_atom[a];
    for (var axis = 0u; axis < 3u; axis++) {
      velocities[atom * 3u + axis] = component(state[a], axis);
    }
  }
}
