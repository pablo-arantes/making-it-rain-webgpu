/**
 * The whole force field, in double precision.
 *
 * Until now the reference was used force by force, which is what the
 * differential tests need: a disagreement in `HarmonicAngleForce` should not be
 * reported as a disagreement in "the energy". The minimiser needs the opposite
 * -- one function from coordinates to an energy and a gradient -- and so does
 * any check that the device's *total* force is right rather than each of its
 * parts.
 *
 * It composes the existing pieces and adds nothing of its own except the
 * dispatch, so the thresholds every part was validated against still apply.
 * The three terms the kernels leave to the host (the dispersion correction, the
 * Ewald self energy and the net-charge correction) are included here, because a
 * minimiser that saw a different energy from the one being reported would be
 * minimising something nobody is looking at -- though none of them depends on a
 * coordinate at fixed volume, so none contributes a force.
 *
 * This is deliberately the slow path. It evaluates PME on a mesh in f64 and
 * walks a cell list in TypeScript; on 26,749 atoms that is seconds per call,
 * which is fine for an oracle and hopeless for a run.
 */

import type { NonbondedForce, OpenMMSystem, Vec3 } from "../engine/system.js";
import { ewaldAlpha, pmeMinimumGridSize, smoothFftSize } from "../engine/units.js";
import { harmonicAngle, harmonicBond, periodicTorsion } from "./forces/bonded.js";
import {
  directSpace,
  netChargeCorrection,
  rectangularBox,
  selfEnergy,
} from "./forces/nonbonded.js";
import { reciprocalSpace } from "./forces/pme.js";
import { positionRestraint } from "./forces/restraint.js";

export interface PotentialResult {
  /** kJ/mol. */
  readonly energy: number;
  /** kJ/mol/nm, xyz interleaved. */
  readonly forces: Float64Array;
  /** Per force type, for a report that needs to say where a difference is. */
  readonly byForce: Readonly<Record<string, number>>;
}

export interface PotentialOptions {
  /**
   * The PME mesh.
   *
   * Left out, it is the smooth size the engine would choose, which is what a
   * comparison against the engine wants. OpenMM's Reference platform uses the
   * exact minimum instead, which is often not smooth, so a comparison against
   * *that* has to pass it explicitly -- see src/reference/forces/pme.ts.
   */
  readonly grid?: Vec3;
}

/** The nonbonded force a system carries, if it has one. */
export const nonbondedForce = (
  system: OpenMMSystem,
): NonbondedForce | undefined =>
  system.forces.find((f): f is NonbondedForce => f.type === "NonbondedForce");

/** The Ewald splitting parameter a force implies, honouring an explicit alpha. */
export const splittingParameter = (force: NonbondedForce): number =>
  force.alpha !== 0 ? force.alpha : ewaldAlpha(force.cutoff, force.ewaldTolerance);

/** The mesh the engine would use for a box. */
export const defaultGrid = (
  force: NonbondedForce,
  box: Vec3,
  alpha: number,
): Vec3 =>
  box.map((extent) =>
    smoothFftSize(pmeMinimumGridSize(alpha, extent, force.ewaldTolerance)),
  ) as unknown as Vec3;

export function evaluatePotential(
  system: OpenMMSystem,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  options: PotentialOptions = {},
): PotentialResult {
  const forces = new Float64Array(system.particleCount * 3);
  const byForce: Record<string, number> = {};
  let energy = 0;
  const add = (name: string, value: number): void => {
    byForce[name] = (byForce[name] ?? 0) + value;
    energy += value;
  };

  for (const force of system.forces) {
    switch (force.type) {
      case "HarmonicBondForce":
        add(force.type, harmonicBond(force, positions, forces));
        break;
      case "HarmonicAngleForce":
        add(force.type, harmonicAngle(force, positions, forces));
        break;
      case "PeriodicTorsionForce":
        add(force.type, periodicTorsion(force, positions, forces));
        break;
      case "NonbondedForce": {
        const box = rectangularBox(boxVectors);
        const alpha = splittingParameter(force);
        const direct = directSpace(force, positions, boxVectors, alpha, forces);
        add("NonbondedForce[direct]", direct.total);
        const grid = options.grid ?? defaultGrid(force, box, alpha);
        const reciprocal = reciprocalSpace(
          force,
          positions,
          boxVectors,
          alpha,
          grid,
          forces,
        );
        const volume = (box[0] as number) * (box[1] as number) * (box[2] as number);
        // The self energy and the net-charge term are constants of the
        // parameters and the volume, and OpenMM reports both in the reciprocal
        // group. They move the energy and not the forces.
        add(
          "NonbondedForce[reciprocal]",
          reciprocal.energy +
            selfEnergy(force, alpha) +
            netChargeCorrection(force, alpha, volume),
        );
        break;
      }
      case "PositionRestraint":
        add(force.type, positionRestraint(force, positions, boxVectors, forces));
        break;
      case "MonteCarloBarostat":
        // Not a force. It rescales the box from `updateContextState`, and
        // contributes no energy of its own.
        break;
      case "CMMotionRemover":
        // Not a force. It acts on velocities, from `updateContextState`.
        break;
      default: {
        // Exhaustive: every member of `Force` is handled above, and
        // `parseSystem` refuses anything outside that union. The `never`
        // binding is what makes adding a force type a compile error here
        // rather than a silently missing energy term.
        const unhandled: never = force;
        throw new Error(
          `evaluatePotential does not implement ${
            (unhandled as { type: string }).type
          }. The engine only claims the forces this workflow produces; an ` +
            "uploaded system with anything else has to be rejected rather " +
            "than silently mis-evaluated.",
        );
      }
    }
  }
  return { energy, forces, byForce };
}
