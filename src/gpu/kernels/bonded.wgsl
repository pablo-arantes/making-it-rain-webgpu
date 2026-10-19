// Bonded forces: harmonic bonds, harmonic angles, periodic torsions.
//
// Three entry points sharing one accumulation path.  Each term is handled by
// one invocation, which scatters its contribution to the two to four atoms it
// touches through the fixed-point atomics of fixed64.wgsl.
//
// Scatter, not gather, and the reason is worth stating because docs/PLAN.md
// originally argued the other way.  The plan preferred a gather kernel over a
// full neighbour list -- at twice the pair work -- partly to keep the
// determinism invariant.  That argument does not apply: fixed-point
// accumulation is *integer* addition, which is exact and commutative, so the
// total does not depend on the order the GPU ran the atomics in.  A scattered
// atomic accumulation is bit-for-bit reproducible. Phase 0 also measured the
// atomics as cheap on this adapter (374 million adds a second into a single
// contended slot).  Gather remains worth measuring for the nonbonded kernel,
// where the work saved is different, but it buys no determinism anywhere.
//
// Positions arrive in fixed point, two words per component, 24 bytes an atom.
// Differences are taken in fixed point -- where they are exact -- and only then
// converted to f32, which is what AGENTS.md requires: a bond vector is at most
// a few tenths of a nanometre, where f32 has plenty of room, while an absolute
// coordinate converted to f32 would throw away the reason positions are stored
// this way.

@group(0) @binding(0) var<storage, read>       positions: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       term_atoms: array<u32>;
@group(0) @binding(2) var<storage, read>       term_params: array<f32>;
@group(0) @binding(3) var<storage, read_write> force: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> energy: array<atomic<u32>>;
@group(0) @binding(5) var<uniform>             params: Params;

struct Params {
  term_count: u32,
}

/// Add a 64-bit fixed-point value into one accumulator slot.
///
/// The force buffer holds two words per slot, low then high, so that both live
/// in one binding. `atomicAdd` returns the value the slot held before the add,
/// which is how the carry out of the low word is detected without a 64-bit
/// atomic: unsigned addition wrapped exactly when the sum is smaller than what
/// was there. Other invocations adding in between do not matter, because the
/// total carry is the sum of the per-operation carries.
fn accumulate(buffer_index: u32, words: vec2<u32>) {
  let slot = buffer_index * 2u;
  let previous = atomicAdd(&force[slot], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&force[slot + 1u], 1u);
  }
  atomicAdd(&force[slot + 1u], words.y);
}

fn accumulate_energy(value: f32) {
  let words = to_fixed(value);
  let previous = atomicAdd(&energy[0], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&energy[1], 1u);
  }
  atomicAdd(&energy[1], words.y);
}

/// Add a force vector to one atom.
fn add_force(atom: u32, f: vec3<f32>) {
  accumulate(atom * 3u + 0u, to_fixed(f.x));
  accumulate(atom * 3u + 1u, to_fixed(f.y));
  accumulate(atom * 3u + 2u, to_fixed(f.z));
}

/// The vector from atom `a` to atom `b`, differenced in fixed point.
fn delta(a: u32, b: u32) -> vec3<f32> {
  return vec3<f32>(
    from_fixed(sub64(positions[b * 3u + 0u], positions[a * 3u + 0u])),
    from_fixed(sub64(positions[b * 3u + 1u], positions[a * 3u + 1u])),
    from_fixed(sub64(positions[b * 3u + 2u], positions[a * 3u + 2u])),
  );
}

// E = (1/2) k (r - r0)^2
@compute @workgroup_size(64)
fn harmonic_bond(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.term_count) { return; }
  let a = term_atoms[i * 2u + 0u];
  let b = term_atoms[i * 2u + 1u];
  let r0 = term_params[i * 2u + 0u];
  let k = term_params[i * 2u + 1u];

  let d = delta(a, b);
  let r = length(d);
  let dr = r - r0;
  accumulate_energy(0.5 * k * dr * dr);
  // The force on `a` points along (b - a) when the bond is stretched.
  let scale = k * dr / r;
  add_force(a, scale * d);
  add_force(b, -scale * d);
}

// E = (1/2) k (theta - theta0)^2
@compute @workgroup_size(64)
fn harmonic_angle(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.term_count) { return; }
  let a = term_atoms[i * 3u + 0u];
  let b = term_atoms[i * 3u + 1u];
  let c = term_atoms[i * 3u + 2u];
  let theta0 = term_params[i * 2u + 0u];
  let k = term_params[i * 2u + 1u];

  // Both bond vectors point away from the central atom.
  let v1 = delta(b, a);
  let v2 = delta(b, c);
  let r1sq = dot(v1, v1);
  let r2sq = dot(v2, v2);
  let cosine = clamp(dot(v1, v2) / sqrt(r1sq * r2sq), -1.0, 1.0);
  let theta = acos(cosine);
  let d_theta = theta - theta0;
  accumulate_energy(0.5 * k * d_theta * d_theta);

  // n is perpendicular to the angle's plane, with |n| = r1 r2 sin(theta). A
  // collinear angle has no plane and no restoring direction; OpenMM floors the
  // same quantity at 1e-6 rather than dividing by zero.
  let n = cross(v1, v2);
  let n_norm = max(length(n), 1e-6);
  let d_e_d_theta = k * d_theta;
  let f1 = -(d_e_d_theta / (r1sq * n_norm)) * cross(v1, n);
  let f3 = (d_e_d_theta / (r2sq * n_norm)) * cross(v2, n);
  add_force(a, f1);
  add_force(c, f3);
  // The central atom takes whatever keeps the triple force-free.
  add_force(b, -f1 - f3);
}

// E = k (1 + cos(periodicity * phi - phase))
@compute @workgroup_size(64)
fn periodic_torsion(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.term_count) { return; }
  let a = term_atoms[i * 4u + 0u];
  let b = term_atoms[i * 4u + 1u];
  let c = term_atoms[i * 4u + 2u];
  let d = term_atoms[i * 4u + 3u];
  let periodicity = term_params[i * 3u + 0u];
  let phase = term_params[i * 3u + 1u];
  let k = term_params[i * 3u + 2u];

  let b1 = delta(a, b);
  let b2 = delta(b, c);
  let b3 = delta(c, d);
  let n1 = cross(b1, b2);
  let n2 = cross(b2, b3);
  let n1sq = dot(n1, n1);
  let n2sq = dot(n2, n2);
  let b2_norm = length(b2);

  // atan2 rather than acos: defined over the whole circle, and it does not lose
  // precision where the two planes are nearly parallel.
  let phi = atan2(b2_norm * dot(b1, n2), dot(n1, n2));
  accumulate_energy(k * (1.0 + cos(periodicity * phi - phase)));
  if (n1sq == 0.0 || n2sq == 0.0) { return; }

  let d_e_d_phi = -k * periodicity * sin(periodicity * phi - phase);
  // dphi/dp1 = -|b2| n1 / |n1|^2 and dphi/dp4 = +|b2| n2 / |n2|^2 for the phi
  // above. Both were confirmed against a central difference of phi rather than
  // taken from a textbook: the signs depend on how b1, b2 and b3 are defined
  // and on the argument order inside atan2.
  let f1 = (d_e_d_phi * b2_norm / n1sq) * n1;
  let f4 = (-d_e_d_phi * b2_norm / n2sq) * n2;
  // How far along b2 each end's lever arm projects decides the inner split.
  let b2sq = b2_norm * b2_norm;
  let s = dot(b1, b2) / b2sq;
  let t = dot(b3, b2) / b2sq;
  add_force(a, f1);
  add_force(b, -(1.0 + s) * f1 + t * f4);
  add_force(c, s * f1 - (1.0 + t) * f4);
  add_force(d, f4);
}
