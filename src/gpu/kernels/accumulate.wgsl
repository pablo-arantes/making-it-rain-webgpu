// Force and energy accumulation, and position differencing.
//
// Shared by every force kernel, which fixes the first three bindings for all of
// them: positions, forces, energy.  A kernel's own data starts at binding 3 and
// its uniform is binding 5.
//
// Forces and energy both accumulate in the 64-bit fixed point of fixed64.wgsl,
// two words per slot, low then high, so a scattered atomic accumulation is
// exact and its total does not depend on the order the adds landed in.

@group(0) @binding(0) var<storage, read>       positions: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> force: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> energy: array<atomic<u32>>;

/// Add a 64-bit fixed-point value into one slot of a two-word accumulator.
///
/// `atomicAdd` returns the value the slot held before the add, which is how the
/// carry out of the low word is detected without a 64-bit atomic: unsigned
/// addition wrapped exactly when the sum is smaller than what was there. Other
/// invocations adding in between do not matter, because the total carry is the
/// sum of the per-operation carries.
fn accumulate_into(slot: u32, words: vec2<u32>) {
  let previous = atomicAdd(&force[slot * 2u], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&force[slot * 2u + 1u], 1u);
  }
  atomicAdd(&force[slot * 2u + 1u], words.y);
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
  accumulate_into(atom * 3u + 0u, to_fixed(f.x));
  accumulate_into(atom * 3u + 1u, to_fixed(f.y));
  accumulate_into(atom * 3u + 2u, to_fixed(f.z));
}

/// The vector from atom `a` to atom `b`, differenced in fixed point.
///
/// Exact in fixed point and only then converted: a bond or pair vector is at
/// most the cutoff, where f32 has room to spare, while converting an absolute
/// coordinate would throw away the reason positions are stored this way.
fn delta(a: u32, b: u32) -> vec3<f32> {
  return vec3<f32>(
    from_fixed(sub64(positions[b * 3u + 0u], positions[a * 3u + 0u])),
    from_fixed(sub64(positions[b * 3u + 1u], positions[a * 3u + 1u])),
    from_fixed(sub64(positions[b * 3u + 2u], positions[a * 3u + 2u])),
  );
}

/// The same, wrapped to the nearest periodic image of a rectangular box.
fn delta_periodic(a: u32, b: u32, box: vec3<f32>) -> vec3<f32> {
  let d = delta(a, b);
  return d - box * round(d / box);
}
