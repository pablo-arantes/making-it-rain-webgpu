/**
 * A mixed-radix Stockham FFT, and the plan the GPU kernel executes.
 *
 * This is deliberately not just "an FFT": it is a model of the kernel. The
 * stage list, the index arithmetic and the twiddle angles here are the ones
 * `src/gpu/kernels/fft.wgsl` uses, so the two can be compared stage by stage
 * rather than only end to end. `src/reference/dft.ts`, the direct O(n^2)
 * transform, is what *this* is checked against -- an independent implementation
 * one level further down.
 *
 * Stockham rather than in-place Cooley-Tukey, because Stockham auto-sorts:
 * each stage reads one buffer and writes the other with a different access
 * pattern, and there is no digit-reversal permutation to get wrong. Two buffers
 * and a ping-pong is a small price for deleting a whole class of bug.
 *
 * Radices 2, 3, 5 and 7, which covers every 2/3/5/7-smooth size -- the sizes
 * `smoothFftSize` rounds a PME grid up to. 25 is 5x5 and 54 is 2x3x3x3, which
 * are the two the committed fixtures need.
 */

/** The radices the transform can use, tried largest first. */
export const RADICES = [7, 5, 3, 2] as const;

export class NotSmooth extends Error {
  constructor(readonly length: number, readonly remainder: number) {
    super(
      `${length} is not 2/3/5/7-smooth (${remainder} is left over after ` +
        "factoring), so this transform cannot do it. PME grid sizes come from " +
        "smoothFftSize, which only returns smooth sizes.",
    );
    this.name = "NotSmooth";
  }
}

/** One Stockham stage: a radix, and how many points are already combined. */
export interface FftStage {
  readonly radix: number;
  /** `Ns`: the product of the radices of all earlier stages. */
  readonly ns: number;
}

export interface FftPlan {
  readonly length: number;
  readonly stages: readonly FftStage[];
}

/**
 * Factor a length into stages.
 *
 * Largest radix first, which keeps the stage count down: 54 becomes
 * [3, 3, 3, 2] rather than [2, 3, 3, 3], and each stage is a full pass over the
 * grid, so fewer is better.
 */
export function planFft(length: number): FftPlan {
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError(`transform length must be a positive integer, got ${length}`);
  }
  const stages: FftStage[] = [];
  let remaining = length;
  let ns = 1;
  for (const radix of RADICES) {
    while (remaining % radix === 0) {
      stages.push({ radix, ns });
      ns *= radix;
      remaining /= radix;
    }
  }
  if (remaining !== 1) throw new NotSmooth(length, remaining);
  return { length, stages };
}

/**
 * One Stockham stage over a set of lines, out of place.
 *
 * Mirrors the kernel exactly, including the index arithmetic:
 *
 *   base   = (line / stride) * (stride * length) + (line % stride)
 *   gather   in[j + r * (length / radix)]
 *   twiddle  exp(sign * 2 pi i * (j % ns) * r / (ns * radix))
 *   scatter  out[(j / ns) * ns * radix + (j % ns) + r * ns]
 *
 * The one `base` formula serves all three axes of a 3D transform: with stride 1
 * it walks z, with stride nz it walks y, with stride ny*nz it walks x.
 */
/**
 * `exp(sign * 2 pi i * j / length)` for every `j`, rounded once to f32.
 *
 * 🔴 **The kernel reads its twiddle factors from this rather than calling
 * `cos` and `sin`, and that is a correctness decision, not an optimisation.**
 * WGSL places no accuracy bound on either function, so a conforming backend may
 * be arbitrarily bad: SwiftShader computes them to **1,587 ulps**, which took
 * the mixed-radix transform from 1.4e-6 to 3.3e-4 relative RMS while Metal,
 * whose `cos` is within 0.77 of an ulp, stayed accurate. The angle was measured
 * at the same time and exonerated -- 5.95e-7 radians against Metal's 7.21e-7 --
 * so the transcendental was the whole of it.
 * `docs/measurements/fft-twiddle-precision.json`.
 *
 * Computed here in f64, where `Math.cos` is accurate to well under an ulp of
 * f64, and rounded once on the store into `Float32Array`. The error that
 * survives is that single rounding, about 6e-8, which is the floor for an f32
 * table and two orders below the 1e-5 the transform is held to.
 *
 * It also takes two transcendentals out of the kernel's innermost block, which
 * `CLAUDE.md` warns is the most expensive place in the engine to put anything.
 *
 * The kernel indexes this by integer arithmetic alone:
 *
 *   gather      `(j % ns) * r * (length / (ns * radix))`
 *   butterfly   `((k * m) % radix) * (length / radix)`
 *
 * Both are exact because `ns * radix` and `radix` divide `length` at every
 * stage, being prefix products of the same radices; `test/fft.test.ts` asserts
 * that those indices reproduce the angles `fftStage` uses.
 */
export function twiddleTable(length: number, sign: number): Float32Array {
  if (!Number.isInteger(length) || length < 1) {
    throw new RangeError(`table length must be a positive integer, got ${length}`);
  }
  const table = new Float32Array(length * 2);
  for (let j = 0; j < length; j++) {
    const angle = (sign * 2 * Math.PI * j) / length;
    table[2 * j] = Math.cos(angle);
    table[2 * j + 1] = Math.sin(angle);
  }
  return table;
}

export function fftStage(
  input: Float64Array,
  output: Float64Array,
  stage: FftStage,
  length: number,
  stride: number,
  lineCount: number,
  sign: number,
): void {
  const { radix, ns } = stage;
  const butterflies = length / radix;
  const values = new Float64Array(radix * 2);
  for (let line = 0; line < lineCount; line++) {
    const base = Math.floor(line / stride) * (stride * length) + (line % stride);
    for (let j = 0; j < butterflies; j++) {
      const twiddleIndex = j % ns;
      // Gather, applying the twiddle on the way in.
      for (let r = 0; r < radix; r++) {
        const at = (base + (j + r * butterflies) * stride) * 2;
        const re = input[at] as number;
        const im = input[at + 1] as number;
        if (r === 0 || twiddleIndex === 0) {
          values[r * 2] = re;
          values[r * 2 + 1] = im;
        } else {
          const angle = (sign * 2 * Math.PI * twiddleIndex * r) / (ns * radix);
          const c = Math.cos(angle);
          const s = Math.sin(angle);
          values[r * 2] = re * c - im * s;
          values[r * 2 + 1] = re * s + im * c;
        }
      }
      // The radix-point transform, by definition. The radix is at most 7, so
      // this is at most 49 terms and needs no specialised butterfly to be fast
      // enough to be worth checking a specialised one against.
      const destination = Math.floor(j / ns) * ns * radix + twiddleIndex;
      for (let k = 0; k < radix; k++) {
        let re = 0;
        let im = 0;
        for (let m = 0; m < radix; m++) {
          const angle = (sign * 2 * Math.PI * k * m) / radix;
          const c = Math.cos(angle);
          const s = Math.sin(angle);
          const vr = values[m * 2] as number;
          const vi = values[m * 2 + 1] as number;
          re += vr * c - vi * s;
          im += vr * s + vi * c;
        }
        const at = (base + (destination + k * ns) * stride) * 2;
        output[at] = re;
        output[at + 1] = im;
      }
    }
  }
}

/**
 * Transform every line along one axis, ping-ponging between two buffers.
 *
 * Returns whichever buffer holds the result, which is `a` when the stage count
 * is even and `b` when it is odd.
 */
export function fftAxis(
  a: Float64Array,
  b: Float64Array,
  plan: FftPlan,
  stride: number,
  lineCount: number,
  sign: number,
): Float64Array {
  let from = a;
  let to = b;
  for (const stage of plan.stages) {
    fftStage(from, to, stage, plan.length, stride, lineCount, sign);
    const swap = from;
    from = to;
    to = swap;
  }
  return from;
}

/**
 * A full 3D transform, unnormalised, in the same convention as
 * `src/reference/dft.ts`: `sign` is -1 forward and +1 inverse, and a forward
 * followed by an inverse multiplies by the number of points.
 *
 * PME needs no normalisation anywhere, because the forward and inverse
 * transforms each appear once in an expression that is quadratic in the grid.
 */
export function fft3d(
  data: Float64Array,
  size: readonly [number, number, number],
  sign: number,
  scratch?: Float64Array,
): Float64Array {
  const [nx, ny, nz] = size;
  const total = nx * ny * nz;
  if (data.length !== total * 2) {
    throw new RangeError(`expected ${total * 2} values for ${nx}x${ny}x${nz}`);
  }
  const other = scratch ?? new Float64Array(total * 2);
  let current = data;
  let spare = other;
  for (const [length, stride] of [
    [nz, 1],
    [ny, nz],
    [nx, ny * nz],
  ] as const) {
    const plan = planFft(length);
    const result = fftAxis(current, spare, plan, stride, total / length, sign);
    if (result !== current) {
      // An odd stage count left the answer in the other buffer; keep going from
      // there rather than copying.
      const swap = current;
      current = result;
      spare = swap;
    }
  }
  if (current !== data) data.set(current);
  return data;
}
