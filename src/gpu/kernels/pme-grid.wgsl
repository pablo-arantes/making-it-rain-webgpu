// Moving the charge grid between its fixed-point and complex forms, and the
// reciprocal-space convolution.

@group(0) @binding(0) var<storage, read>       fixed_grid: array<u32>;
@group(0) @binding(1) var<storage, read_write> complex_grid: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             size: Size;

struct Size {
  point_count: u32,
}

/// Fixed-point real charge grid -> complex f32 grid, ready for the transform.
@compute @workgroup_size(64)
fn load_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= size.point_count) { return; }
  let words = vec2<u32>(fixed_grid[i * 2u], fixed_grid[i * 2u + 1u]);
  complex_grid[i] = vec2<f32>(from_fixed(words), 0.0);
}
