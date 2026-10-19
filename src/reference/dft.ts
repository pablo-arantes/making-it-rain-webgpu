/**
 * A 3D discrete Fourier transform, by three passes of one-dimensional
 * transforms.
 *
 * Each 1D transform is the O(n^2) definition rather than a fast algorithm, and
 * that is a deliberate choice for the reference: the PME grids this engine
 * uses are 25 to 64 points an axis, so a 3D transform is a few times 1e7
 * complex operations -- about a second in JavaScript -- and a direct
 * transform has no twiddle tables, no radix cases and no bit reversal to get
 * wrong.  `src/reference/` exists to be obviously correct, not fast.
 *
 * A fast transform belongs in the WGSL kernel, where it is measured against
 * this one.
 */

export interface ComplexGrid {
  readonly size: readonly [number, number, number];
  /** Interleaved real, imaginary; index (((x * ny) + y) * nz + z) * 2. */
  readonly data: Float64Array;
}

export function complexGrid(size: readonly [number, number, number]): ComplexGrid {
  return { size, data: new Float64Array(size[0] * size[1] * size[2] * 2) };
}

/** The twiddle factors for one axis length, shared across its lines. */
function twiddles(n: number, sign: number): { cos: Float64Array; sin: Float64Array } {
  const cos = new Float64Array(n * n);
  const sin = new Float64Array(n * n);
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      const angle = (sign * 2 * Math.PI * k * j) / n;
      cos[k * n + j] = Math.cos(angle);
      sin[k * n + j] = Math.sin(angle);
    }
  }
  return { cos, sin };
}

/**
 * Transform in place along one axis.
 *
 * `sign` is -1 for the forward transform and +1 for the inverse, matching the
 * usual physics convention; neither direction is normalised, and PME needs no
 * normalisation because the forward and inverse transforms appear once each in
 * an expression that is quadratic in the grid.
 */
function transformAxis(grid: ComplexGrid, axis: 0 | 1 | 2, sign: number): void {
  const [nx, ny, nz] = grid.size;
  const n = grid.size[axis] as number;
  const { cos, sin } = twiddles(n, sign);
  const stride = axis === 0 ? ny * nz : axis === 1 ? nz : 1;
  const line = new Float64Array(n * 2);
  const result = new Float64Array(n * 2);

  const outerA = axis === 0 ? ny : nx;
  const outerB = axis === 2 ? ny : nz;
  for (let a = 0; a < outerA; a++) {
    for (let b = 0; b < outerB; b++) {
      // The flat index of the first element of this line.
      let base: number;
      if (axis === 0) base = (a * nz + b) as number;
      else if (axis === 1) base = (a * ny * nz + b) as number;
      else base = (a * ny + b) * nz;

      for (let i = 0; i < n; i++) {
        const at = (base + i * stride) * 2;
        line[i * 2] = grid.data[at] as number;
        line[i * 2 + 1] = grid.data[at + 1] as number;
      }
      for (let k = 0; k < n; k++) {
        let real = 0;
        let imaginary = 0;
        const row = k * n;
        for (let j = 0; j < n; j++) {
          const c = cos[row + j] as number;
          const s = sin[row + j] as number;
          const re = line[j * 2] as number;
          const im = line[j * 2 + 1] as number;
          real += re * c - im * s;
          imaginary += re * s + im * c;
        }
        result[k * 2] = real;
        result[k * 2 + 1] = imaginary;
      }
      for (let i = 0; i < n; i++) {
        const at = (base + i * stride) * 2;
        grid.data[at] = result[i * 2] as number;
        grid.data[at + 1] = result[i * 2 + 1] as number;
      }
    }
  }
}

/** Forward 3D transform, in place, unnormalised. */
export function forward3d(grid: ComplexGrid): void {
  transformAxis(grid, 0, -1);
  transformAxis(grid, 1, -1);
  transformAxis(grid, 2, -1);
}

/** Inverse 3D transform, in place, unnormalised. */
export function inverse3d(grid: ComplexGrid): void {
  transformAxis(grid, 0, 1);
  transformAxis(grid, 1, 1);
  transformAxis(grid, 2, 1);
}
