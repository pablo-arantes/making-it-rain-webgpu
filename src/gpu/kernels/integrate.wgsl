// The two integrators, one thread per particle.
//
// Transcribed from src/reference/integrator.ts, which was itself transcribed
// from `ReferenceVerletDynamics::update` and
// `ReferenceLangevinMiddleDynamics::update`. Each integrator is split at the
// points where the constraint solver has to run, because that is a separate
// dispatch: the host interleaves them, and the split points are the contract.
//
//   Verlet    verlet_kick_drift -> SHAKE(positions, trial) -> verlet_finish
//   LFMiddle  lfmiddle_kick -> RATTLE(positions, velocities)
//             -> lfmiddle_drift -> SHAKE(positions, trial) -> lfmiddle_finish
//
// Three things about the number formats here, all of them deliberate.
//
// **Positions are fixed point, velocities are f32.** This looks inconsistent
// and is not. What forces fixed point on a position is the ratio between the
// coordinate and its increment: a 6.5 nm coordinate moves by about 1e-7 nm in a
// 2 fs step, a ratio of 6.5e7, while f32 resolves 1.7e7 -- so the increment
// does not round, it vanishes. A velocity is about 1 nm/ps and its increment
// about 1e-3 nm/ps, a ratio of 1e3, which f32 has four orders of magnitude of
// room for. OpenMM's own mixed precision draws the line in exactly this place,
// keeping `posq` in double-single and `velm` in float4.
//
// **The drift carries a residual.** `to_fixed` rounds, and the part that did
// not fit is added back into the next step's displacement rather than dropped.
// Dropping it is not a random error -- a particle drifting steadily rounds the
// same way every step -- so it accumulates linearly in the step count. With
// feedback the outstanding residual stays bounded by one unit in the last
// place forever. OpenMM does not do this; it is a deviation, and a strictly
// conservative one. See docs/DEVIATIONS.md and src/gpu/kernels/position.wgsl,
// where it was measured at 2.3e-5 nm over 400 ps without.
//
// **Masses arrive as f32**, so a particle's response differs from the f64
// reference by about 6e-8 relative. That is the mixed-precision contract, and
// it is equivalent to simulating with masses perturbed far below the precision
// they are known to.

@group(0) @binding(0) var<storage, read_write> positions: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<f32>;
@group(0) @binding(2) var<storage, read>       forces: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> trial: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> unconstrained: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> residual: array<f32>;
@group(0) @binding(6) var<storage, read>       inverse_mass: array<f32>;
@group(0) @binding(7) var<uniform>             params: StepParams;

struct StepParams {
  particle_count: u32,
  seed: u32,
  step: u32,
  // Bit 0: draw thermostat noise. Zero friction leaves `noise_scale` at zero,
  // and then the draw is skipped entirely rather than multiplied away -- which
  // is what makes a friction-free run reproducible without a seed.
  flags: u32,
  time_step: f32,
  // exp(-dt * friction), and sqrt(1 - that squared).
  velocity_scale: f32,
  noise_scale: f32,
  // Boltzmann's constant times the temperature, kJ/mol.
  kt: f32,
}

const DRAW_NOISE: u32 = 1u;

/// Displace a fixed-point coordinate by `wanted` nm, feeding back what did not
/// fit. Returns the new coordinate; `left_over` carries the remainder out.
fn displace(from_position: vec2<u32>, wanted: f32, left_over: ptr<function, f32>) -> vec2<u32> {
  let taken = to_fixed(wanted);
  *left_over = wanted - from_fixed(taken);
  return add64(from_position, taken);
}

/// One full-step kick: v += dt * F / m.
fn kick(particle: u32, inverse: f32) {
  let dt = params.time_step;
  for (var axis = 0u; axis < 3u; axis++) {
    let at = particle * 3u + axis;
    velocities[at] = velocities[at] + dt * inverse * from_fixed(forces[at]);
  }
}

// ---------------------------------------------------------------------------
// Velocity Verlet
// ---------------------------------------------------------------------------

/// Kick, then drift a whole step into `trial`.
///
/// A frozen particle gets `trial = position` so that the constraint solver,
/// which reads `trial` for every atom of a cluster, sees a consistent set.
@compute @workgroup_size(64)
fn verlet_kick_drift(@builtin(global_invocation_id) gid: vec3<u32>) {
  let particle = gid.x;
  if (particle >= params.particle_count) { return; }
  let inverse = inverse_mass[particle];
  if (inverse == 0.0) {
    for (var axis = 0u; axis < 3u; axis++) {
      let at = particle * 3u + axis;
      trial[at] = positions[at];
    }
    return;
  }
  kick(particle, inverse);
  let dt = params.time_step;
  for (var axis = 0u; axis < 3u; axis++) {
    let at = particle * 3u + axis;
    var left_over = 0.0;
    trial[at] = displace(
      positions[at],
      velocities[at] * dt + residual[at],
      &left_over,
    );
    residual[at] = left_over;
  }
}

/// The constrained displacement *is* the velocity: v = (x' - x) / dt.
///
/// This is what folds the constraint's effect back in, and it is why the
/// difference is taken in fixed point and only then converted -- the
/// displacement is tiny and the coordinates it came from are not.
@compute @workgroup_size(64)
fn verlet_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
  let particle = gid.x;
  if (particle >= params.particle_count) { return; }
  if (inverse_mass[particle] == 0.0) { return; }
  let scale = 1.0 / params.time_step;
  for (var axis = 0u; axis < 3u; axis++) {
    let at = particle * 3u + axis;
    velocities[at] = scale * from_fixed(sub64(trial[at], positions[at]));
    positions[at] = trial[at];
  }
}

// ---------------------------------------------------------------------------
// LFMiddle -- what `LangevinIntegrator` has been since OpenMM 8.2
// ---------------------------------------------------------------------------

/// Kick only. The host then RATTLEs the velocities, which Verlet does not do.
@compute @workgroup_size(64)
fn lfmiddle_kick(@builtin(global_invocation_id) gid: vec3<u32>) {
  let particle = gid.x;
  if (particle >= params.particle_count) { return; }
  let inverse = inverse_mass[particle];
  if (inverse == 0.0) { return; }
  kick(particle, inverse);
}

/// Half drift, thermostat, half drift -- the "middle" the scheme is named for.
///
/// `unconstrained` keeps a copy of the trial positions before the solver moves
/// them, which is OpenMM's `oldx`: the velocity correction in `lfmiddle_finish`
/// is measured against it, not against the starting positions.
@compute @workgroup_size(64)
fn lfmiddle_drift(@builtin(global_invocation_id) gid: vec3<u32>) {
  let particle = gid.x;
  if (particle >= params.particle_count) { return; }
  let inverse = inverse_mass[particle];
  if (inverse == 0.0) {
    for (var axis = 0u; axis < 3u; axis++) {
      let at = particle * 3u + axis;
      trial[at] = positions[at];
      unconstrained[at] = positions[at];
    }
    return;
  }

  var deviate = array<f32, 3>(0.0, 0.0, 0.0);
  if ((params.flags & DRAW_NOISE) != 0u) {
    let drawn = thermostat_normals(params.seed, params.step, particle);
    deviate = array<f32, 3>(drawn.x, drawn.y, drawn.z);
  }
  // The noise has the amplitude that makes the stationary distribution the
  // right temperature for *this* particle's mass.
  let noise = params.noise_scale * sqrt(params.kt * inverse);
  let half_step = 0.5 * params.time_step;

  for (var axis = 0u; axis < 3u; axis++) {
    let at = particle * 3u + axis;
    var left_over = residual[at];
    var moved = displace(positions[at], velocities[at] * half_step + left_over, &left_over);
    let thermostatted = params.velocity_scale * velocities[at] + noise * deviate[axis];
    velocities[at] = thermostatted;
    moved = displace(moved, thermostatted * half_step + left_over, &left_over);
    residual[at] = left_over;
    trial[at] = moved;
    unconstrained[at] = moved;
  }
}

/// Add the velocity the constraint projection implies, and commit the move.
@compute @workgroup_size(64)
fn lfmiddle_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
  let particle = gid.x;
  if (particle >= params.particle_count) { return; }
  if (inverse_mass[particle] == 0.0) { return; }
  let scale = 1.0 / params.time_step;
  for (var axis = 0u; axis < 3u; axis++) {
    let at = particle * 3u + axis;
    velocities[at] =
      velocities[at] + scale * from_fixed(sub64(trial[at], unconstrained[at]));
    positions[at] = trial[at];
  }
}
