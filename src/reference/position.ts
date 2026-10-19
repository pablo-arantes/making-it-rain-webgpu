/**
 * The reference for position integration in 64-bit fixed point.
 *
 * Positions use the same format as forces, for the same reason, plus one more.
 * The shared reason is exactness: integer addition cannot lose a bit and does
 * not depend on the order the additions happen in.  The extra reason is that
 * no optimiser can algebraically simplify an integer add, and on Chrome/Metal
 * one can and does simplify the floating-point alternative into uselessness --
 * see docs/PRECISION.md.
 *
 * Resolution 2^-32 nm, about 2.3e-10 nm, against the 1e-7 nm that the
 * notebook's 1e-6 constraint tolerance implies on an X-H bond.  Range +/-2^31
 * nm, against the 6.52 nm box of the largest committed fixture.
 */

import { FIXED_SCALE, fromFixed, fromFixedF32, toFixed } from "./fixed-point.js";

/**
 * Plain fixed-point accumulation: convert each displacement and add it.
 *
 * The conversion error is at most half the format's resolution, but for a
 * repeated displacement it is the *same* error every step, so it accumulates
 * linearly in the step count rather than random-walking.  Measured: 200000
 * displacements of 1e-7 nm drift 2.3e-5 nm.  `integrateCompensated` is what
 * the engine uses.
 */
export function integrate(startNm: number, displacements: ArrayLike<number>): bigint {
  let position = toFixed(Math.fround(startNm));
  for (let i = 0; i < displacements.length; i++) {
    position += toFixed(Math.fround(displacements[i] as number));
  }
  return position;
}

/**
 * Fixed-point accumulation carrying the part of each displacement that did not
 * fit the format into the next step.
 *
 * The outstanding residual stays bounded by one unit in the last place, so the
 * drift of a long run stops growing with the step count.  Mirrors the
 * `compensated` accumulator in src/gpu/kernels/position.wgsl step for step.
 */
export function integrateCompensated(
  startNm: number,
  displacements: ArrayLike<number>,
): { position: bigint; residualNm: number } {
  let position = toFixed(Math.fround(startNm));
  let residual = 0;
  for (let i = 0; i < displacements.length; i++) {
    const wanted = Math.fround(Math.fround(displacements[i] as number) + residual);
    const taken = toFixed(wanted);
    position += taken;
    // `fromFixedF32`, not `fromFixed`: the kernel computes this in f32, and the
    // residual it leaves behind has to be the same f32 on both sides.
    residual = Math.fround(wanted - fromFixedF32(taken));
  }
  return { position, residualNm: residual };
}

/** `integrate`, as nanometres. */
export const integrateNm = (
  startNm: number,
  displacements: ArrayLike<number>,
): number => fromFixed(integrate(startNm, displacements));

/** `integrateCompensated`, as nanometres. */
export const integrateCompensatedNm = (
  startNm: number,
  displacements: ArrayLike<number>,
): number => fromFixed(integrateCompensated(startNm, displacements).position);

/** Reassemble the two words the GPU wrote for one coordinate. */
export function positionFromWords(lo: number, hi: number): bigint {
  return BigInt(hi | 0) * FIXED_SCALE + BigInt(lo >>> 0);
}

/**
 * The displacement between two fixed-point positions, in nm.
 *
 * This is how the engine will read positions: take the difference in fixed
 * point, where it is exact, and only then convert to f32.  A pair vector is at
 * most the cutoff, where f32's ulp is around 6e-8 nm, so the conversion costs
 * nothing that matters.  Converting an *absolute* position to f32 would throw
 * away exactly what this representation exists to keep.
 */
export const displacementNm = (a: bigint, b: bigint): number => fromFixed(a - b);
