/**
 * Kernel sources, and the one-line include mechanism WGSL does not have.
 *
 * `composeKernel` prepends shared WGSL to a kernel so there is exactly one copy
 * of the fixed-point arithmetic, the accumulation and the erfc approximation,
 * rather than a copy per kernel that can drift from the reference
 * implementation.
 *
 * `accumulate` also fixes the first three bindings for every force kernel --
 * positions, forces, energy -- so a kernel's own data starts at binding 3 and
 * its uniform is binding 5. That convention is what lets `dispatch` pass
 * bindings positionally.
 */

import fixed64 from "./kernels/fixed64.wgsl?raw";
import accumulate from "./kernels/accumulate.wgsl?raw";
import erfc from "./kernels/erfc.wgsl?raw";
import fixedPointAccumulate from "./kernels/fixed-point.wgsl?raw";
import position from "./kernels/position.wgsl?raw";
import bonded from "./kernels/bonded.wgsl?raw";
import nonbondedPairs from "./kernels/nonbonded-pairs.wgsl?raw";
import nonbondedExceptions from "./kernels/nonbonded-exceptions.wgsl?raw";
import fft from "./kernels/fft.wgsl?raw";
import pmeSpread from "./kernels/pme.wgsl?raw";
import pmeGrid from "./kernels/pme-grid.wgsl?raw";
import pmeConvolve from "./kernels/pme-convolve.wgsl?raw";
import pmeInterpolate from "./kernels/pme-interpolate.wgsl?raw";
import settle from "./kernels/settle.wgsl?raw";
import shake from "./kernels/shake.wgsl?raw";

export const SHARED = { fixed64, accumulate, erfc } as const;

export function composeKernel(
  source: string,
  includes: readonly (keyof typeof SHARED)[] = ["fixed64"],
): string {
  const prelude = includes.map((name) => SHARED[name]).join("\n");
  return `${prelude}\n// ---- kernel ----\n${source}`;
}

/** Force accumulation on its own: contended atomic adds in fixed point. */
export const FIXED_POINT_ACCUMULATE = composeKernel(fixedPointAccumulate);

/** Position integration in fixed point, plus the Dekker diagnostic. */
export const POSITION = composeKernel(position);

/**
 * Bonded forces. Three entry points -- `harmonic_bond`, `harmonic_angle` and
 * `periodic_torsion` -- sharing one fixed-point accumulation path.
 */
export const BONDED = composeKernel(bonded, ["fixed64", "accumulate"]);

/** Real-space Ewald and Lennard-Jones over the pairs inside the cutoff. */
export const NONBONDED_PAIRS = composeKernel(nonbondedPairs, [
  "fixed64",
  "accumulate",
  "erfc",
]);

/** The exclusion corrections and the exceptions' own interactions. */
export const NONBONDED_EXCEPTIONS = composeKernel(nonbondedExceptions, [
  "fixed64",
  "accumulate",
  "erfc",
]);

/**
 * One mixed-radix Stockham FFT stage. Needs no shared include: it is pure
 * complex f32 arithmetic with no fixed point anywhere.
 */
export const FFT = composeKernel(fft, []);

/** B-spline charge spreading onto the mesh. */
export const PME_SPREAD = composeKernel(pmeSpread);

/** Fixed-point charge grid to complex f32 grid. */
export const PME_GRID = composeKernel(pmeGrid);

/** The reciprocal-space convolution and the mesh energy. */
export const PME_CONVOLVE = composeKernel(pmeConvolve);

/** Interpolating the convolved potential back onto the atoms as forces. */
export const PME_INTERPOLATE = composeKernel(pmeInterpolate);

/** SETTLE, analytic, one thread per rigid three-atom cluster. */
export const SETTLE = composeKernel(settle);

/** SHAKE and RATTLE, iterative, one thread per cluster. */
export const SHAKE = composeKernel(shake);
