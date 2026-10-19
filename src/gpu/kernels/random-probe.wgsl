// A probe that exposes the generator's output so the host can compare it.
//
// Two separate outputs, because they fail for different reasons and conflating
// them wastes a debugging cycle:
//
//   * `bits` is the raw Threefry output, which is integer arithmetic and must
//     be bit-identical to the host's. A difference here is a wrong rotation
//     table or a missed key injection.
//   * `deviates` is what Box-Muller makes of it, where the host computes in
//     f64 and the device in f32 with `log`, `sqrt` and `cos` specified only to
//     a few ulp. A difference here is arithmetic, not logic.

@group(0) @binding(0) var<storage, read_write> bits: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> deviates: array<f32>;
@group(0) @binding(2) var<uniform>             params: ProbeParams;

struct ProbeParams {
  particle_count: u32,
  seed: u32,
  step: u32,
  pad: u32,
}

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let particle = gid.x;
  if (particle >= params.particle_count) { return; }
  let key = vec2<u32>(params.seed, THERMOSTAT_DOMAIN);
  bits[particle * 2u + 0u] = threefry2x32(vec2<u32>(particle * 2u, params.step), key);
  bits[particle * 2u + 1u] = threefry2x32(vec2<u32>(particle * 2u + 1u, params.step), key);
  let drawn = thermostat_normals(params.seed, params.step, particle);
  deviates[particle * 3u + 0u] = drawn.x;
  deviates[particle * 3u + 1u] = drawn.y;
  deviates[particle * 3u + 2u] = drawn.z;
}
