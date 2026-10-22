/**
 * Position restraints, in double precision.
 *
 * The notebook's equilibration adds a `CustomExternalForce` holding
 * `k*periodicdistance(x, y, z, x0, y0, z0)^2` with per-particle `k, x0, y0, z0`,
 * which pins chosen atoms near where they started while the solvent relaxes
 * around them. The engine implements that one expression directly rather than
 * through an expression parser, as `docs/PLAN.md` specifies, and
 * `parseSystem` rejects a `CustomExternalForce` carrying anything else --
 * evaluating the wrong energy quietly is worse than refusing.
 *
 * `periodicdistance` is the minimum-image distance, so the energy is
 *
 *     E = sum over restrained i of k_i * |wrap(r_i - r0_i)|^2
 *
 * and the force is `-2 k_i wrap(r_i - r0_i)`. The wrap is what makes this a
 * restraint rather than a tether to a particular periodic image: an atom that
 * crosses the boundary is pulled towards the nearest copy of its reference
 * point, not dragged back across the box.
 *
 * 🔴 The wrap is also the one place this can go wrong invisibly. Without it the
 * energy is identical for every atom that has not crossed a boundary, which on
 * a freshly minimised structure is all of them -- so an implementation missing
 * the minimum image passes a single-point energy test and then applies forces
 * of order `k * boxLength` to whichever atom drifts across first. On the
 * notebook's `k` that is 10,000 kJ/mol/nm^2 times 3 nm.
 */

import type { PositionRestraintForce, Vec3 } from "../../engine/system.js";
import { minimumImage, reduceBox, isReducedForm, type BoxVectors } from "../triclinic.js";

/**
 * The restraint energy, accumulating forces into `forces` when given.
 *
 * Returns kJ/mol. `forces` is in kJ/mol/nm, xyz interleaved, and is added to
 * rather than overwritten, so several forces can share one accumulator.
 */
export function positionRestraint(
  force: PositionRestraintForce,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  forces?: Float64Array,
): number {
  // 🔴 No cutoff bound here, unlike the nonbonded forces: a restraint's
  // displacement is meant to be small -- it is how far an atom has drifted from
  // where it is held -- so the nearest image is the whole answer and there is
  // no cutoff to exceed. `minimumImage` handles a skewed cell; on a rectangular
  // one it is the same arithmetic per axis that was here before.
  const box: BoxVectors = isReducedForm(boxVectors as BoxVectors)
    ? (boxVectors as BoxVectors)
    : reduceBox(boxVectors as BoxVectors);
  let energy = 0;
  for (let term = 0; term < force.count; term++) {
    const particle = force.particle[term] as number;
    const k = force.k[term] as number;
    const reference = [
      force.x0[term] as number,
      force.y0[term] as number,
      force.z0[term] as number,
    ];
    // Nearest image. `Math.round` resolves a tie upward inside `minimumImage`
    // and OpenMM's `floor(x + 0.5)` does the same, so the two agree on the
    // boundary.
    const image = minimumImage(
      [reference[0] as number, reference[1] as number, reference[2] as number],
      [
        positions[particle * 3] as number,
        positions[particle * 3 + 1] as number,
        positions[particle * 3 + 2] as number,
      ],
      box,
    );
    const delta = [image[0] as number, image[1] as number, image[2] as number];
    const squared =
      (delta[0] as number) ** 2 + (delta[1] as number) ** 2 + (delta[2] as number) ** 2;
    energy += k * squared;
    if (forces !== undefined) {
      for (let axis = 0; axis < 3; axis++) {
        forces[particle * 3 + axis] =
          (forces[particle * 3 + axis] as number) - 2 * k * (delta[axis] as number);
      }
    }
  }
  return energy;
}
