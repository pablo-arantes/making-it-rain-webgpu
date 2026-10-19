/**
 * Kernel sources, and the one-line include mechanism WGSL does not have.
 *
 * `composeKernel` prepends shared WGSL to a kernel so there is exactly one
 * copy of the fixed-point arithmetic rather than a copy per kernel that can
 * drift from the reference implementation.
 */

import fixed64 from "./kernels/fixed64.wgsl?raw";
import fixedPointAccumulate from "./kernels/fixed-point.wgsl?raw";
import position from "./kernels/position.wgsl?raw";

export const SHARED = { fixed64 } as const;

export function composeKernel(
  source: string,
  includes: readonly (keyof typeof SHARED)[] = ["fixed64"],
): string {
  const prelude = includes.map((name) => SHARED[name]).join("\n");
  return `${prelude}\n// ---- kernel ----\n${source}`;
}

/** Force accumulation: contended atomic adds in fixed point. */
export const FIXED_POINT_ACCUMULATE = composeKernel(fixedPointAccumulate);

/** Position integration in fixed point, plus the Dekker diagnostic. */
export const POSITION = composeKernel(position);
