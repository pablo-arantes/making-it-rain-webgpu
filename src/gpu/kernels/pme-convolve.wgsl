// The reciprocal-space convolution, and the mesh energy.
//
//   E = prefactor * sum_{k != 0} exp(-k^2 / 4 alpha^2) / (k^2 |b(m)|^2) |Q(m)|^2
//
// with prefactor = 2 pi / V * ONE_4PI_EPS0. The same expression, in the same
// form, as `reciprocalSpace` in src/reference/forces/pme.ts; see
// docs/NONBONDED.md for where it came from and what else the reciprocal force
// group contains.
//
// The grid is overwritten with twice the convolved value, because that is
// dE/dQ: E is quadratic in the grid, so the interpolation that follows needs
// the factor of two. Leaving it out made every reciprocal force exactly half
// what it should be, which the reference's numerical derivative caught.

@group(0) @binding(0) var<storage, read_write> grid: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       moduli: array<f32>;
@group(0) @binding(2) var<storage, read_write> energy: array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             convolution: Convolution;

struct Convolution {
  grid_size: vec3<u32>,
  /// Where the y and z moduli start in `moduli`; the x ones start at zero.
  modulus_offset: vec2<u32>,
  box: vec3<f32>,
  alpha: f32,
  prefactor: f32,
}

const TWO_PI: f32 = 6.283185307179586;

@compute @workgroup_size(64)
fn convolve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = convolution.grid_size;
  let total = n.x * n.y * n.z;
  let index = gid.x;
  if (index >= total) { return; }

  let mz = index % n.z;
  let my = (index / n.z) % n.y;
  let mx = index / (n.z * n.y);

  if (mx == 0u && my == 0u && mz == 0u) {
    // The k = 0 term is the neutralising background, dropped here under the
    // usual tin-foil boundary condition. What a non-neutral system needs
    // instead is a closed-form correction; see netChargeCorrection.
    grid[index] = vec2<f32>(0.0, 0.0);
    return;
  }

  // Frequencies above the midpoint are the negative ones.
  let signed = vec3<i32>(
    select(i32(mx) - i32(n.x), i32(mx), mx * 2u <= n.x),
    select(i32(my) - i32(n.y), i32(my), my * 2u <= n.y),
    select(i32(mz) - i32(n.z), i32(mz), mz * 2u <= n.z),
  );
  let k = TWO_PI * vec3<f32>(signed) / convolution.box;
  let k_squared = dot(k, k);
  let modulus = moduli[mx]
              * moduli[convolution.modulus_offset.x + my]
              * moduli[convolution.modulus_offset.y + mz];

  let factor = convolution.prefactor
             * exp(-k_squared / (4.0 * convolution.alpha * convolution.alpha))
             / (k_squared * modulus);
  let value = grid[index];
  let contribution = factor * dot(value, value);

  // Energy accumulates in fixed point, so the total does not depend on which
  // grid point landed first.
  let words = to_fixed(contribution);
  let previous = atomicAdd(&energy[0], words.x);
  if (previous + words.x < previous) {
    atomicAdd(&energy[1], 1u);
  }
  atomicAdd(&energy[1], words.y);

  grid[index] = 2.0 * factor * value;
}
