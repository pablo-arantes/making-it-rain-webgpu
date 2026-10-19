// Zero a range of words on the device.
//
// Every step starts by clearing the force and energy accumulators and the PME
// charge grid. Doing that with `writeBuffer` means the host allocates and sends
// a zeroed array each time -- 642 KB a step for 3HTB's forces alone -- and the
// copy has to be ordered against the previous step's reads. A dispatch is one
// more pass in a list the loop already encodes, and it moves no data at all.
//
// Deliberately untyped: it zeroes `u32`, which is the representation of a zero
// f32 and of a zero fixed-point pair as well, so one kernel serves the force
// accumulator, the energy, and the complex grids.

@group(0) @binding(0) var<storage, read_write> words: array<u32>;
@group(0) @binding(1) var<uniform>             extent: vec4<u32>;

@compute @workgroup_size(64)
fn zero_words(@builtin(global_invocation_id) gid: vec3<u32>) {
  // `extent.x` is the word count. A uniform block has to be at least 16 bytes
  // and a vec4 says so in the type rather than in a comment about padding.
  if (gid.x >= extent.x) { return; }
  words[gid.x] = 0u;
}
