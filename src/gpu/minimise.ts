/**
 * Minimisation with the forces on the device.
 *
 * The search itself stays on the host, which is not a shortcut: OpenMM's own
 * minimiser works this way, setting a context's positions and asking for forces
 * and energy on every evaluation. L-BFGS is a few vector operations per
 * iteration over 3N numbers, against a force evaluation that is forty-odd
 * dispatches, so the arithmetic is not where the time goes. What it costs is one
 * position upload and two readbacks per evaluation -- about 1.9 MB each way for
 * 3HTB -- and a readback drains the pipeline. That is the price of being able to
 * run the identical algorithm the reference runs, including the constraint
 * springs and their schedule.
 *
 * 🔴 The positions are fixed point, with a resolution of 2^-32 nm. A backtracking
 * line search halves its step until the energy improves, and once a trial step
 * falls below that resolution the device sees no movement at all, so the search
 * stalls where a double-precision one would keep going. That is a real limit of
 * the format and not a bug; `src/gpu/validate-minimise.ts` measures where it
 * bites rather than leaving it to be discovered.
 */

import type { OpenMMSystem } from "../engine/system.js";
import { minimise, type MinimiseOptions, type MinimiseResult } from "../reference/minimise.js";
import type { StepLoop } from "./loop.js";

/**
 * Minimise the loop's current configuration, leaving it in place.
 *
 * The loop's coordinates are the minimiser's starting point and are overwritten
 * with the result, so a caller can minimise and then run without copying
 * anything back and forth.
 */
export async function minimiseOnDevice(
  loop: StepLoop,
  system: OpenMMSystem,
  options: Omit<MinimiseOptions, "constraintTolerance"> &
    Partial<Pick<MinimiseOptions, "constraintTolerance">> = {},
): Promise<MinimiseResult> {
  const start = await loop.positions();
  const result = await minimise(
    system,
    start,
    (positions) => loop.potential(positions),
    {
      ...options,
      constraintTolerance:
        options.constraintTolerance ?? loop.options.constraintTolerance,
    },
  );
  // The last evaluation was at whatever the line search tried last, which is
  // not necessarily where the minimiser ended up.
  loop.setPositions(result.positions);
  return result;
}
