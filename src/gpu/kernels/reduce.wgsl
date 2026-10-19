// Whole-system reductions: centre-of-mass momentum and kinetic energy.
//
// Both sum one number over every particle, and both are summed in the same
// 64-bit fixed point the forces use, for the same reason: the total must not
// depend on the order the partial sums landed in. A floating-point reduction
// over 26,749 terms gives a slightly different answer each time the scheduler
// interleaves the workgroups differently, and a centre-of-mass correction that
// differs run to run makes a trajectory irreproducible -- which would quietly
// undo the work of making everything else deterministic.
//
// Two levels: each workgroup reduces into workgroup-local fixed-point
// accumulators, then one invocation folds that into the global total. 418
// global atomics for a 26,749-atom system instead of 80,247.
//
// Exactness applies to the *sum*, not to the terms. Each term is a product in
// f32, and the masses themselves arrive as f32, so a term carries about 6e-8
// relative error. Summing 26,749 of those with random signs leaves the
// centre-of-mass velocity about 6e-10 nm/ps from the f64 reference's -- below
// what the thermostat does to it in a single step. OpenMM's GPU platforms
// reduce this quantity in floating point, so this is the stricter of the two.

@group(0) @binding(0) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(1) var<storage, read>       mass: array<f32>;
@group(0) @binding(2) var<storage, read_write> totals: array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             params: ReduceParams;

struct ReduceParams {
  particle_count: u32,
  pad: u32,
  /// One over the sum of every mass, including frozen particles' zeros.
  inverse_total_mass: f32,
  pad2: f32,
}

/// Three axes, two words each. The kinetic energy uses only the first slot.
var<workgroup> partial: array<atomic<u32>, 6>;

fn fold_local(slot: u32, words: vec2<u32>) {
  let previous = atomicAdd(&partial[slot * 2u], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&partial[slot * 2u + 1u], 1u);
  }
  atomicAdd(&partial[slot * 2u + 1u], words.y);
}

fn fold_global(slot: u32) {
  let low = atomicLoad(&partial[slot * 2u]);
  let high = atomicLoad(&partial[slot * 2u + 1u]);
  let previous = atomicAdd(&totals[slot * 2u], low);
  if (previous + low < previous) {
    atomicAdd(&totals[slot * 2u + 1u], 1u);
  }
  atomicAdd(&totals[slot * 2u + 1u], high);
}

fn clear_partials(local_index: u32) {
  if (local_index < 6u) { atomicStore(&partial[local_index], 0u); }
}

/// The mass-weighted velocity sum, per axis, into totals[0..6].
@compute @workgroup_size(64)
fn momentum(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  clear_partials(local_index);
  workgroupBarrier();
  let particle = gid.x;
  if (particle < params.particle_count) {
    // A frozen particle has mass zero, so it contributes nothing here and is
    // left out of the total mass as well -- which is what keeps its velocity
    // untouched by the correction.
    let m = mass[particle];
    for (var axis = 0u; axis < 3u; axis++) {
      fold_local(axis, to_fixed(m * velocities[particle * 3u + axis]));
    }
  }
  workgroupBarrier();
  if (local_index == 0u) {
    for (var axis = 0u; axis < 3u; axis++) { fold_global(axis); }
  }
}

/// Subtract the mass-weighted mean velocity, which `momentum` must have put in
/// `totals` first.
///
/// This is `CMMotionRemover`, and it is not a force: OpenMM applies it from
/// `ContextImpl::updateContextState`, at the start of a step and before the
/// forces. Leaving it out slides the whole trajectory while satisfying every
/// constraint -- see src/reference/com-motion.ts, where it cost 1.75e-5 nm a
/// step and two wrong diagnoses.
@compute @workgroup_size(64)
fn subtract_com(@builtin(global_invocation_id) gid: vec3<u32>) {
  let particle = gid.x;
  if (particle >= params.particle_count) { return; }
  if (mass[particle] == 0.0) { return; }
  for (var axis = 0u; axis < 3u; axis++) {
    let total = vec2<u32>(
      atomicLoad(&totals[axis * 2u]),
      atomicLoad(&totals[axis * 2u + 1u]),
    );
    let at = particle * 3u + axis;
    velocities[at] = velocities[at] - from_fixed(total) * params.inverse_total_mass;
  }
}

/// Sum of m v^2 / 2 over every particle, into totals[0..2].
///
/// This is the unshifted convention, which is what `LangevinIntegrator`
/// reports and so what the notebook's log shows. Verlet's conserved quantity
/// uses velocities shifted half a step and re-projected; see
/// src/reference/kinetic.ts and docs/DYNAMICS.md.
@compute @workgroup_size(64)
fn kinetic(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_index) local_index: u32,
) {
  clear_partials(local_index);
  workgroupBarrier();
  let particle = gid.x;
  if (particle < params.particle_count) {
    let v = vec3<f32>(
      velocities[particle * 3u + 0u],
      velocities[particle * 3u + 1u],
      velocities[particle * 3u + 2u],
    );
    fold_local(0u, to_fixed(0.5 * mass[particle] * dot(v, v)));
  }
  workgroupBarrier();
  if (local_index == 0u) { fold_global(0u); }
}
