// Deciding when the Verlet neighbour list has gone stale, on the device.
//
// A list built to `cutoff + skin` is valid for as long as no *pair* has closed
// by more than the skin, and no pair can close by more than the sum of the two
// atoms' displacements. So the list is valid while every atom is within half
// the skin of where it was when the list was built, which is a per-atom test
// and needs no pair work at all.
//
// 🔴 The decision is made here rather than on the host because the host cannot
// see it without a readback, and a readback is a full pipeline drain -- every
// step, to answer a question whose answer is almost always no. Instead the flag
// lives in a buffer, this kernel sets it, and the rebuild passes are encoded
// every step and return immediately when it is clear. A dispatch that does
// nothing costs a launch.
//
// It is also why the flag cannot simply be a step counter. A fixed interval
// would be wrong exactly when the system moves fastest -- a minimiser's first
// steps, a thermostat equilibrating a bad structure -- and wrong silently, by
// losing pairs. `RunOptions.cellListInterval` was that fixed interval, and
// `verletSkin` plus this kernel is what replaced it.
//
//   flags[0]  rebuild: set by `displacement`, read by every rebuild pass
//   flags[1]  built:   0 until the first list exists, so step 0 rebuilds
//   flags[2]  overflow: sticky, an atom had more neighbours than room
//   flags[3]  rebuilds: how many have happened, so the skin can be judged on
//             a measured interval rather than an assumed one
//   flags[4]  the largest displacement-since-build ever seen, in pm, so the
//             skin can be judged against what the atoms actually did

@group(0) @binding(0) var<storage, read>       positions: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> reference_positions: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> flags: array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             params: VerletParams;

struct VerletParams {
  particle_count: u32,
  /// `(skin / 2)^2`. Squared so the test needs no square root.
  half_skin_squared: f32,
}

/// Clear the rebuild flag, or set it if no list has been built yet.
///
/// One invocation. It runs before `displacement`, which only ever ors the flag
/// on, so something has to turn it off once a rebuild has happened.
@compute @workgroup_size(1)
fn reset_rebuild() {
  let built = atomicLoad(&flags[1]);
  atomicStore(&flags[0], select(1u, 0u, built == 1u));
}

/// Set the rebuild flag if any atom has moved half the skin since the build.
///
/// The displacement is taken from the *unwrapped* positions. Wrapped ones jump
/// by a whole box when an atom crosses a face, which would rebuild the list
/// every few steps for no reason.
@compute @workgroup_size(64)
fn displacement(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particle_count) { return; }
  if (atomicLoad(&flags[0]) != 0u) { return; }   // already decided

  // A difference of two positions, which is small, so converting it is safe --
  // unlike converting either one. No minimum image: this is motion, not a pair
  // vector, and an atom that has wandered out of the box has really moved.
  let d = vec3<f32>(
    from_fixed(sub64(positions[i * 3u + 0u], reference_positions[i * 3u + 0u])),
    from_fixed(sub64(positions[i * 3u + 1u], reference_positions[i * 3u + 1u])),
    from_fixed(sub64(positions[i * 3u + 2u], reference_positions[i * 3u + 2u])),
  );
  let moved = dot(d, d);
  // Picometres, which is plenty of resolution for a skin quoted in tenths of a
  // nanometre and keeps this an integer atomic.
  atomicMax(&flags[4], u32(sqrt(moved) * 1000.0));
  if (moved > params.half_skin_squared) {
    atomicOr(&flags[0], 1u);
  }
}

/// Remember where the atoms were, for the next `displacement` to measure from.
@compute @workgroup_size(64)
fn save_reference(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (atomicLoad(&flags[0]) == 0u) { return; }
  let i = gid.x;
  if (i >= params.particle_count) { return; }
  reference_positions[i * 3u + 0u] = positions[i * 3u + 0u];
  reference_positions[i * 3u + 1u] = positions[i * 3u + 1u];
  reference_positions[i * 3u + 2u] = positions[i * 3u + 2u];
}

/// Record that a list now exists.
///
/// A separate dispatch from `save_reference` because the flag must not be
/// cleared until every one of its invocations has copied its atom, and only a
/// pass boundary guarantees that.
@compute @workgroup_size(1)
fn mark_built() {
  if (atomicLoad(&flags[0]) != 0u) {
    atomicAdd(&flags[3], 1u);
  }
  atomicStore(&flags[1], 1u);
}
