/**
 * The neighbour list against the cell-shell scan it replaced.
 *
 * 🔴 This check exists because its absence cost a day. The two paths find the
 * same pairs by two different routes, so one evaluation of each at the same
 * coordinates must agree to the bit -- the arithmetic after a candidate is
 * accepted is literally the same function. Anything else means the list is
 * missing pairs, and a missing pair is not a small error that shows up as a
 * tolerance: it is a discontinuity in the force that integrates into a blown-up
 * trajectory a few hundred steps later.
 *
 * The force comparison against OpenMM did not catch it, and could not: the
 * reference fixtures that exercise the engine's geometry most are small, and a
 * small box gives three cells an axis, where the 27-cell shell *is* the whole
 * box and no search radius can reach outside it. The bug only exists for a box
 * big enough to have cells the shell does not cover. So this runs on whichever
 * fixture it is given, and the one that matters is the big one.
 */

import { StepLoop, type RunOptions } from "./loop.js";
import { parseState } from "../engine/state.js";
import { parseSystem } from "../engine/system.js";

export interface NeighbourComparison {
  readonly particleCount: number;
  readonly capacity: number;
  readonly skin: number;
  readonly cells: readonly number[];
  readonly cellExtent: readonly number[];
  /** `cutoff + skin`, which one cell has to be at least as wide as. */
  readonly searchRadius: number;
  readonly shellEnergy: number;
  readonly listEnergy: number;
  readonly energyRelativeError: number;
  /** The largest single force component difference, kJ/mol/nm. */
  readonly worstForceDifference: number;
  /** How many components differ at all. Zero is the only right answer. */
  readonly differing: number;
  readonly overflowed: boolean;
  readonly ok: boolean;
}

const OPTIONS: RunOptions = {
  timeStep: 0.002,
  temperature: 298,
  friction: 1,
  seed: 24301,
  integrator: "lfmiddle",
  constraintTolerance: 1e-6,
};

export async function validateNeighbours(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
): Promise<NeighbourComparison> {
  const system = parseSystem(texts.systemXml);
  const state = parseState(texts.stateXml);

  // The shell path: no list, every atom scans its 27 cells.
  const shellLoop = await StepLoop.fromParsed(device, system, state, {
    ...OPTIONS,
    verletSkin: 0,
  });
  const shell = await shellLoop.potential(state.positions);
  shellLoop.destroy();

  // The list path, at the same coordinates.
  const listLoop = await StepLoop.fromParsed(device, system, state, OPTIONS);
  const list = await listLoop.potential(state.positions);
  const flags = await listLoop.verletFlags();
  const geometry = listLoop.cellGeometry();
  listLoop.destroy();

  let worst = 0;
  let differing = 0;
  for (let k = 0; k < shell.forces.length; k++) {
    const difference = Math.abs((list.forces[k] as number) - (shell.forces[k] as number));
    if (difference !== 0) differing++;
    if (difference > worst) worst = difference;
  }
  const energyRelativeError =
    shell.energy === 0 ? 0 : Math.abs(list.energy - shell.energy) / Math.abs(shell.energy);

  return {
    particleCount: system.particleCount,
    capacity: flags.capacity,
    skin: flags.skin,
    cells: geometry.divisions,
    cellExtent: geometry.extent,
    searchRadius: system.forces.reduce(
      (radius, force) =>
        force.type === "NonbondedForce" ? force.cutoff + flags.skin : radius,
      0,
    ),
    shellEnergy: shell.energy,
    listEnergy: list.energy,
    energyRelativeError,
    worstForceDifference: worst,
    differing,
    overflowed: flags.overflowed,
    // Bit for bit, because the same candidates go through the same function.
    ok: differing === 0 && !flags.overflowed,
  };
}
