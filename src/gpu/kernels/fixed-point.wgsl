// Force accumulation in 64-bit fixed point.
//
// WebGPU has no f64 and no 64-bit atomics, and atomics exist only on i32 and
// u32.  Accumulating forces in f32 would make the result depend on the order
// the GPU happened to run the threads in, and on this workload that order
// dependence is not small: for the 3HTB system PME's direct space is
// +1.56e6 kJ/mol and its reciprocal space is -1.73e6, cancelling to -1.38e5.
// Each term is more than ten times the answer.
//
// So forces accumulate as integers, in the format defined by fixed64.wgsl,
// which this kernel is composed with.  Integer addition is exact and
// commutative, so the sum does not depend on thread order and nothing is lost
// while accumulating.
//
// Range  +/-2^31 kJ/mol/nm, about +/-2.1e9.  The largest force component
// measured in an unminimised starting structure was 5.6e5, so the headroom is
// about 3800x.

@group(0) @binding(0) var<storage, read>       values: array<f32>;
@group(0) @binding(1) var<storage, read_write> acc_lo: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> acc_hi: array<atomic<i32>>;
@group(0) @binding(3) var<uniform>             params: Params;

struct Params {
  // How many values there are, and how many accumulator slots to spread them
  // over.  Slots far fewer than values is the interesting case: that is what
  // makes the adds contend, which is what this kernel exists to test.
  value_count: u32,
  slot_count: u32,
}

/// Add a fixed-point value into slot `i`, propagating the carry by hand.
///
/// `atomicAdd` returns the value the slot held before the add, so the carry out
/// of the low word is detectable without a 64-bit atomic: unsigned addition
/// wrapped exactly when the sum we wrote is smaller than what was there. Other
/// threads adding in between does not matter, because the total carry is the
/// sum of the per-operation carries.
fn accumulate(i: u32, words: vec2<u32>) {
  let previous = atomicAdd(&acc_lo[i], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&acc_hi[i], 1i);
  }
  atomicAdd(&acc_hi[i], bitcast<i32>(words.y));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= params.value_count) { return; }
  accumulate(v % params.slot_count, to_fixed(values[v]));
}
