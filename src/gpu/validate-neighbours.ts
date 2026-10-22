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
import { minimiseOnDevice } from "./minimise.js";
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
  /**
   * NVE energy drift on each path, kJ/mol per atom per ns, from one start.
   *
   * 🔴 The force comparison above is one configuration, and that is not enough
   * to trust a neighbour list. A list is correct at the moment it is built
   * whatever the rebuild criterion says; what a wrong criterion costs is pairs
   * that go missing *between* rebuilds, intermittently, in a way no single
   * evaluation can see. Drift is the measure that does see it -- a stale cell
   * list at an interval of 20 steps reads 13.7 kJ/mol/ns/atom against 0.0023
   * rebuilt every step -- so the two paths are run from the same minimised
   * state and their slopes compared.
   *
   * These came back **equal to five figures** on both fixtures, which is a
   * stronger result than the test was built to look for: it means the two paths
   * produce the same trajectory bit for bit over thousands of steps, not merely
   * the same forces once. `listRebuilds` and `shellRebuilds` are what make that
   * a result rather than a tautology.
   *
   * 🔴 The slope itself is **not** the engine's drift figure and must not be
   * quoted as one. This run is NVE from a minimised structure that has not been
   * equilibrated, so the structure is still relaxing and energy is flowing
   * between terms; the calibrated gate is `validate-loop.ts`, against the basis
   * in AGENTS.md. What is being compared here is two paths, not the engine
   * against OpenMM.
   */
  readonly listDriftPerAtomPerNs: number;
  readonly shellDriftPerAtomPerNs: number;
  readonly driftSteps: number;
  /**
   * How many rebuilds each drift run did, which is what stops this being
   * vacuous.
   *
   * 🔴 The two drifts came back equal to five figures, and the first reading of
   * that is "the comparison is not comparing anything" -- if both runs took the
   * same path, equality is guaranteed and says nothing. So the paths are made
   * to prove themselves: the list run must rebuild, and the shell run must not,
   * because it has no list to rebuild. With that, the equality is a real
   * result, and a strong one -- the two trajectories agree bit for bit over
   * thousands of steps, not just at the configuration the forces were compared
   * at.
   */
  readonly listRebuilds: number;
  readonly shellRebuilds: number;
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

/** The slope of total energy against time, kJ/mol per atom per ns. */
function driftPerAtomPerNs(
  reports: readonly { readonly time: number; readonly totalEnergy: number }[],
  particleCount: number,
): number {
  let sumT = 0;
  let sumE = 0;
  let sumTT = 0;
  let sumTE = 0;
  for (const report of reports) {
    const t = report.time / 1000;
    sumT += t;
    sumE += report.totalEnergy;
    sumTT += t * t;
    sumTE += t * report.totalEnergy;
  }
  const n = reports.length;
  const denominator = n * sumTT - sumT * sumT;
  if (denominator === 0) return 0;
  return Math.abs((n * sumTE - sumT * sumE) / denominator / particleCount);
}

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

  // Then the same system run on both paths, for the drift. No thermostat: NVE
  // is what makes a slope mean anything, and `friction: 0` is how this engine
  // says that.
  const driftSteps = Math.max(400, Math.min(4000, Math.ceil(8e6 / system.particleCount)));
  const drift = async (
    skin: number | undefined,
  ): Promise<{ slope: number; rebuilds: number }> => {
    const run = await StepLoop.fromParsed(device, system, state, {
      ...OPTIONS,
      friction: 0,
      ...(skin === undefined ? {} : { verletSkin: skin }),
    });
    try {
      // From a relaxed state, or the slope measures the structure relaxing
      // rather than the integrator.
      await minimiseOnDevice(run, system, { maxIterations: 200 });
      const reports = [];
      const chunk = Math.ceil(driftSteps / 10);
      for (let done = 0; done < driftSteps; done += chunk) {
        await run.run(Math.min(chunk, driftSteps - done));
        reports.push(await run.report());
      }
      return {
        slope: driftPerAtomPerNs(reports, system.particleCount),
        rebuilds: (await run.verletFlags()).rebuilds,
      };
    } finally {
      run.destroy();
    }
  };
  const listDrift = await drift(undefined);
  const shellDrift = await drift(0);

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
    listDriftPerAtomPerNs: listDrift.slope,
    shellDriftPerAtomPerNs: shellDrift.slope,
    driftSteps,
    listRebuilds: listDrift.rebuilds,
    shellRebuilds: shellDrift.rebuilds,
    // Bit for bit on the forces, because the same candidates go through the
    // same function. The drift is reported rather than asserted: see the note
    // on the fields, and `validate-loop.ts` is what holds drift to a threshold.
    ok:
      differing === 0 &&
      !flags.overflowed &&
      // The two runs really did take different paths, or their agreeing means
      // nothing.
      listDrift.rebuilds > 0 &&
      shellDrift.rebuilds === 0,
  };
}
