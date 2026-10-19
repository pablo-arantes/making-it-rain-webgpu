// Reciprocal-space PME: spreading, loading, convolution and interpolation.
//
// Four entry points, each with its own bindings, because `layout: "auto"`
// derives a layout from what an entry point actually uses.
//
// The charge grid is spread with fixed-point atomics, like every other
// accumulation in this engine: an atom touches 5^3 = 125 grid points and
// neighbouring atoms touch the same ones, so this is the densest scatter in the
// whole workload, and integer accumulation makes it exact and order-independent
// anyway. See docs/PRECISION.md.

@group(0) @binding(0) var<storage, read>       pme_positions: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read>       pme_charge: array<f32>;
@group(0) @binding(2) var<storage, read_write> charge_grid: array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             pme: Pme;

struct Pme {
  particle_count: u32,
  grid: vec3<u32>,
  box: vec3<f32>,
  alpha: f32,
  /// 2 pi / V * ONE_4PI_EPS0, the whole constant in front of the mesh sum.
  prefactor: f32,
}

const PME_ORDER: u32 = 5u;

/// The order-5 B-spline weights and their derivatives at offset `w`.
///
/// Mirrors `bsplineWeightsAndDerivatives` in src/reference/forces/bspline.ts,
/// including the recursion order, so the two agree term by term. The weights
/// apply to grid points `floor(u) - order + 1 + k`; that offset is a global
/// shift of the mesh, which becomes a phase in the transform and cancels in
/// |Q(m)|^2.
fn bspline(w: f32) -> array<vec2<f32>, 5> {
  // One level short of the full order, which is also the derivative's source:
  // dM_p/du = M_{p-1}(u) - M_{p-1}(u-1).
  var lower: array<f32, 5>;
  lower[0] = 1.0 - w;
  lower[1] = w;
  lower[2] = 0.0;
  lower[3] = 0.0;
  lower[4] = 0.0;
  for (var k = 3u; k <= PME_ORDER - 1u; k++) {
    let div = 1.0 / f32(k - 1u);
    lower[k - 1u] = div * w * lower[k - 2u];
    for (var j = 1u; j < k - 1u; j++) {
      lower[k - j - 1u] = div * ((w + f32(j)) * lower[k - j - 2u]
                               + (f32(k - j) - w) * lower[k - j - 1u]);
    }
    lower[0] = div * (1.0 - w) * lower[0];
  }
  var derivatives: array<f32, 5>;
  derivatives[0] = -lower[0];
  for (var j = 1u; j < PME_ORDER; j++) {
    derivatives[j] = lower[j - 1u] - lower[j];
  }
  // The final recursion level gives the weights themselves.
  var weights = lower;
  let div = 1.0 / f32(PME_ORDER - 1u);
  weights[PME_ORDER - 1u] = div * w * weights[PME_ORDER - 2u];
  for (var j = 1u; j < PME_ORDER - 1u; j++) {
    weights[PME_ORDER - j - 1u] = div * ((w + f32(j)) * weights[PME_ORDER - j - 2u]
                                       + (f32(PME_ORDER - j) - w) * weights[PME_ORDER - j - 1u]);
  }
  weights[0] = div * (1.0 - w) * weights[0];

  var out: array<vec2<f32>, 5>;
  for (var k = 0u; k < PME_ORDER; k++) {
    out[k] = vec2<f32>(weights[k], derivatives[k]);
  }
  return out;
}

/// Where an atom sits on the mesh: the first grid index it touches, and the
/// fractional offset within its cell.
fn mesh_position(atom: u32, axis: u32, extent: f32, divisions: u32) -> vec2<f32> {
  let coordinate = from_fixed(pme_positions[atom * 3u + axis]);
  let fraction = coordinate / extent - floor(coordinate / extent);
  let scaled = fraction * f32(divisions);
  let cell = floor(scaled);
  return vec2<f32>(cell, scaled - cell);
}

fn grid_add(index: u32, value: f32) {
  let words = to_fixed(value);
  let previous = atomicAdd(&charge_grid[index * 2u], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&charge_grid[index * 2u + 1u], 1u);
  }
  atomicAdd(&charge_grid[index * 2u + 1u], words.y);
}

@compute @workgroup_size(64)
fn spread_charge(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= pme.particle_count) { return; }
  let charge = pme_charge[i];
  if (charge == 0.0) { return; }

  let mx = mesh_position(i, 0u, pme.box.x, pme.grid.x);
  let my = mesh_position(i, 1u, pme.box.y, pme.grid.y);
  let mz = mesh_position(i, 2u, pme.box.z, pme.grid.z);
  let wx = bspline(mx.y);
  let wy = bspline(my.y);
  let wz = bspline(mz.y);
  // floor(u) - order + 1, before wrapping.
  let bx = i32(mx.x) - i32(PME_ORDER) + 1;
  let by = i32(my.x) - i32(PME_ORDER) + 1;
  let bz = i32(mz.x) - i32(PME_ORDER) + 1;

  for (var a = 0u; a < PME_ORDER; a++) {
    let gx = u32((bx + i32(a) + i32(pme.grid.x)) % i32(pme.grid.x));
    let cx = charge * wx[a].x;
    for (var b = 0u; b < PME_ORDER; b++) {
      let gy = u32((by + i32(b) + i32(pme.grid.y)) % i32(pme.grid.y));
      let cxy = cx * wy[b].x;
      for (var c = 0u; c < PME_ORDER; c++) {
        let gz = u32((bz + i32(c) + i32(pme.grid.z)) % i32(pme.grid.z));
        grid_add((gx * pme.grid.y + gy) * pme.grid.z + gz, cxy * wz[c].x);
      }
    }
  }
}
