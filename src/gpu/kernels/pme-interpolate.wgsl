// Interpolating the convolved potential back onto the atoms as forces.
//
// One invocation per atom, gathering: it reads the 5^3 grid points it spread
// onto and sums the gradient, so no atomics are needed for the pair of
// quantities it owns.
//
//   F_i = -q_i * sum_grid (dW/dr_i) * dE/dQ(n)
//
// where dE/dQ is what the convolution left in the grid and dW/dr is the
// B-spline derivative scaled by grid points per nanometre.

@group(0) @binding(0) var<storage, read>       pme_positions: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> force: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read>       pme_charge: array<f32>;
@group(0) @binding(3) var<storage, read>       potential: array<vec2<f32>>;
@group(0) @binding(4) var<uniform>             pme: Pme;

struct Pme {
  particle_count: u32,
  grid: vec3<u32>,
  box: vec3<f32>,
  /// The inverse box, row-major, three `vec4`-strided rows. Diagonal for a
  /// rectangular cell, appended so no existing offset moves.
  inverse: array<vec4<f32>, 3>,
}

const PME_ORDER: u32 = 5u;

fn bspline(w: f32) -> array<vec2<f32>, 5> {
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

/// This atom's fractional coordinate, wrapped into [0, 1).
///
/// 🔴 `f = r · B⁻¹`, which mixes all three Cartesian components -- so unlike
/// the rectangular `coordinate / extent` it replaces, this cannot be done one
/// axis at a time. It is why the charge grid is laid out fractionally at all:
/// `exp(ik·r)` is `exp(2πi m·f)` by construction, so the transform is the same
/// transform whatever the cell shape and only the reciprocal vectors change.
/// For a rectangular cell the inverse is diagonal and this is `r_u / L_u`.
fn fractional_position(atom: u32, inverse: array<vec4<f32>, 3>) -> vec3<f32> {
  let r = vec3<f32>(
    from_fixed(pme_positions[atom * 3u + 0u]),
    from_fixed(pme_positions[atom * 3u + 1u]),
    from_fixed(pme_positions[atom * 3u + 2u]),
  );
  let f = vec3<f32>(
    r.x * inverse[0].x + r.y * inverse[1].x + r.z * inverse[2].x,
    r.x * inverse[0].y + r.y * inverse[1].y + r.z * inverse[2].y,
    r.x * inverse[0].z + r.y * inverse[1].z + r.z * inverse[2].z,
  );
  return f - floor(f);
}

/// The grid cell and offset for one axis, from an already-fractional coordinate.
fn mesh_from_fraction(fraction: f32, divisions: u32) -> vec2<f32> {
  let scaled = fraction * f32(divisions);
  let cell = floor(scaled);
  return vec2<f32>(cell, scaled - cell);
}

fn force_add(slot: u32, value: f32) {
  let words = to_fixed(value);
  let previous = atomicAdd(&force[slot * 2u], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&force[slot * 2u + 1u], 1u);
  }
  atomicAdd(&force[slot * 2u + 1u], words.y);
}

@compute @workgroup_size(64)
fn interpolate_force(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= pme.particle_count) { return; }
  let charge = pme_charge[i];
  if (charge == 0.0) { return; }

  let fraction = fractional_position(i, pme.inverse);
  let mx = mesh_from_fraction(fraction.x, pme.grid.x);
  let my = mesh_from_fraction(fraction.y, pme.grid.y);
  let mz = mesh_from_fraction(fraction.z, pme.grid.z);
  let wx = bspline(mx.y);
  let wy = bspline(my.y);
  let wz = bspline(mz.y);
  let bx = i32(mx.x) - i32(PME_ORDER) + 1;
  let by = i32(my.x) - i32(PME_ORDER) + 1;
  let bz = i32(mz.x) - i32(PME_ORDER) + 1;

  var gradient = vec3<f32>(0.0);
  for (var a = 0u; a < PME_ORDER; a++) {
    let gx = u32((bx + i32(a) + i32(pme.grid.x)) % i32(pme.grid.x));
    for (var b = 0u; b < PME_ORDER; b++) {
      let gy = u32((by + i32(b) + i32(pme.grid.y)) % i32(pme.grid.y));
      for (var c = 0u; c < PME_ORDER; c++) {
        let gz = u32((bz + i32(c) + i32(pme.grid.z)) % i32(pme.grid.z));
        // The convolution left dE/dQ in the real part.
        let value = potential[(gx * pme.grid.y + gy) * pme.grid.z + gz].x;
        gradient += value * vec3<f32>(
          wx[a].y * wy[b].x * wz[c].x,
          wx[a].x * wy[b].y * wz[c].x,
          wx[a].x * wy[b].x * wz[c].y,
        );
      }
    }
  }
  // d/dr = (grid points per nm) * d/dw, and no extra factor: the two the
  // convolution carried is dE/dQ.
  // 🔴 The spline derivatives are with respect to the *fractional* coordinate,
  // so the chain rule brings them back: `dE/dr_v = Σ_u dE/df_u · B⁻¹[v][u]`,
  // which is a row of the inverse dotted with the fractional gradient. The
  // division by the edge length it replaces is what that reduces to when the
  // inverse is diagonal.
  let df = -charge * gradient * vec3<f32>(pme.grid);
  let f = vec3<f32>(
    dot(df, pme.inverse[0].xyz),
    dot(df, pme.inverse[1].xyz),
    dot(df, pme.inverse[2].xyz),
  );
  force_add(i * 3u + 0u, f.x);
  force_add(i * 3u + 1u, f.y);
  force_add(i * 3u + 2u, f.z);
}
