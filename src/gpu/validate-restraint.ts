/**
 * The restraint kernel against OpenMM, at three configurations.
 *
 * The third is the one that matters. At the reference positions the energy is
 * exactly zero and every force vanishes, so that case alone passes an
 * implementation that computes nothing. A small displacement exercises the
 * arithmetic. A displacement of a whole box exercises the **minimum image**,
 * which is the part that can be wrong invisibly -- without it, every atom that
 * has not crossed a boundary gives an identical energy.
 *
 * Expected values come from the restraint's own force group, so they are the
 * restraint alone and not a difference of two large nearly-cancelling totals.
 */

import { parseState, type OpenMMState } from "../engine/state.js";
import { parseSystem, type PositionRestraintForce } from "../engine/system.js";
import { positionRestraint } from "../reference/forces/restraint.js";
import { evaluateRestraint } from "./restraint.js";
import { evaluatePotential } from "../reference/potential.js";
import { StepLoop, type RunOptions } from "./loop.js";

export interface RestraintCase {
  readonly label: string;
  readonly deviceEnergy: number;
  readonly openmmEnergy: number;
  readonly energyError: number;
  /** The worst force-component difference from OpenMM, kJ/mol/nm. */
  readonly worstForceError: number;
  /** And from our own f64 reference, which uses the same algebra. */
  readonly worstForceVsReference: number;
  /** The largest force OpenMM reports, so the errors above have a scale. */
  readonly openmmMaxForce: number;
  readonly ok: boolean;
}

/**
 * The loop's assembled force field against the reference's, on the restrained
 * system.
 *
 * The force is already compared against OpenMM at fixed configurations; what
 * those cannot check is the *wiring* -- that the restraint pass is encoded into
 * a step at all, into the shared accumulator, rather than prepared and
 * forgotten. A restraint that was never dispatched passes every single-point
 * comparison above, because those run the restraint kernel on its own.
 *
 * The first attempt at this measured how far restrained atoms drifted over 200
 * steps against the same atoms unrestrained, and it was a bad test: the ratio
 * came out 1.02, not because the restraint was missing but because the run
 * starts from an un-minimised configuration where the forces driving the motion
 * are ten times the restraint's, and because `k` had been set to 10 instead of
 * the notebook's 700 on an assumption that had never been checked. A
 * statistical claim needs an equilibrated system and a lot of steps.
 *
 * Comparing total energies needs neither -- but not at the wrapped
 * configuration, which was the second wrong turn here. Displacing atoms by 3 nm
 * overlaps them, and the total potential comes out at 3.9e14 kJ/mol where f32
 * and f64 have nothing in common. That configuration isolates the restraint
 * beautifully and is meaningless for a total.
 *
 * The configuration that works is a **rigid translation**. Shifting every atom
 * by the same vector leaves every distance, every angle and every minimum-image
 * pair separation untouched, so the whole force field is unchanged -- while
 * every restrained atom moves exactly that far from its reference point. The
 * restraint's contribution is then known in closed form, `k d^2` per restrained
 * atom, and it is a measurable share of a perfectly ordinary total: 1,078
 * kJ/mol out of -20,000 on 1L2Y at 0.1 nm. It is the same observation as
 * deviation 7 in docs/DEVIATIONS.md, used as an instrument.
 */
export interface AssembledPotential {
  /** kJ/mol, from the loop: every force, as a run would compute it. */
  readonly deviceEnergy: number;
  /** kJ/mol, from the double-precision reference's own assembled potential. */
  readonly referenceEnergy: number;
  readonly relativeError: number;
  /** What the restraint contributes there, so the comparison has a scale. */
  readonly restraintEnergy: number;
  /** `k d^2` per restrained atom: what it must contribute, in closed form. */
  readonly expectedRestraintEnergy: number;
  /** nm, the rigid translation applied. */
  readonly translation: number;
  /** That a short run from the same state stays finite. */
  readonly steps: number;
  readonly finite: boolean;
  readonly ok: boolean;
}

export interface RestraintComparison {
  readonly restrainedAtoms: number;
  readonly cases: readonly RestraintCase[];
  readonly assembled: AssembledPotential | null;
  readonly thresholds: typeof RESTRAINT_THRESHOLDS;
  readonly ok: boolean;
}

/**
 * Thresholds: an absolute floor **or** a relative bound, whichever is met.
 *
 * Neither alone works, because the quantity being checked spans the whole range
 * from exactly zero to ten million. At the reference positions the expected
 * energy and every expected force are exactly zero, where a relative bound is
 * undefined. At 3HTB's wrapped configuration the restraint energy is 9,792,210
 * kJ/mol, where an absolute bound of 1e-2 asks for twelve significant figures
 * from f32 arithmetic -- the measured error there is 0.019 absolute and 1.9e-9
 * relative, and the first version of this file failed it for the wrong reason.
 *
 * The absolute floors are set where the values are small: the displaced case
 * reaches 776 kJ/mol and 205 kJ/mol/nm, so 1e-2 and 1e-3 are parts in 10^5 of
 * those. The relative bound of 1e-6 then covers the large configurations. A
 * missing wrap or a sign error fails both by orders of magnitude.
 */
export const RESTRAINT_THRESHOLDS = {
  energy: 1e-2,
  force: 1e-3,
  relative: 1e-6,
  /**
   * Relative agreement between the loop's assembled energy and the reference's.
   *
   * Loose on purpose: the two differ by every f32-versus-f64 difference in the
   * engine at once, and the direct-space term alone is 3.9e-7 relative on 3HTB.
   * What it has to catch is a missing force, which is percent-scale or worse --
   * the restraint is 2.7% of 1L2Y's total at the wrapped configuration.
   */
  assembledRelative: 1e-4,
} as const;

async function compareAssembledPotential(
  device: GPUDevice,
  restrainedXml: string,
  stateXml: string,
  restraint: PositionRestraintForce,
  translation: number,
  steps: number,
): Promise<AssembledPotential> {
  const system = parseSystem(restrainedXml);
  const state: OpenMMState = parseState(stateXml);
  // Every atom the same distance from its own reference point, with the rest of
  // the force field untouched.
  const positions = Float64Array.from(state.positions);
  for (let i = 0; i < system.particleCount; i++) {
    positions[i * 3] = (positions[i * 3] as number) + translation;
  }
  let expectedRestraintEnergy = 0;
  for (let i = 0; i < restraint.count; i++) {
    expectedRestraintEnergy += (restraint.k[i] as number) * translation * translation;
  }
  const options: RunOptions = {
    timeStep: 0.002,
    temperature: 300,
    friction: 0,
    constraintTolerance: 1e-6,
    integrator: "verlet",
    cellListInterval: 1,
  };
  const loop = await StepLoop.fromParsed(device, system, state, options);
  try {
    const deviceEnergy = (await loop.potential(positions)).energy;
    const reference = evaluatePotential(system, positions, system.boxVectors);
    const referenceEnergy = reference.energy;
    const relativeError =
      Math.abs(deviceEnergy - referenceEnergy) / Math.abs(referenceEnergy);

    // And that a short run from the fixture's own state stays finite, which is
    // the cheapest possible statement that the extra pass did not corrupt the
    // step it was added to.
    loop.setPositions(state.positions);
    await loop.run(steps);
    const after = await loop.positions();
    const finite = after.every((value) => Number.isFinite(value));

    const restraintEnergy = reference.byForce["PositionRestraint"] ?? 0;
    return {
      deviceEnergy,
      referenceEnergy,
      relativeError,
      restraintEnergy,
      expectedRestraintEnergy,
      translation,
      steps,
      finite,
      ok:
        relativeError < RESTRAINT_THRESHOLDS.assembledRelative &&
        finite &&
        // The closed form, as an independent check that the share being
        // compared is the restraint's and not something else's.
        Math.abs(restraintEnergy - expectedRestraintEnergy) <
          1e-6 * Math.abs(expectedRestraintEnergy),
    };
  } finally {
    loop.destroy();
  }
}

export async function validateRestraint(
  device: GPUDevice,
  systemXml: string,
  stateXml: string,
  cases: readonly {
    readonly label: string;
    readonly positions: Float64Array;
    readonly forces: Float64Array;
    readonly energy: number;
  }[],
): Promise<RestraintComparison> {
  const system = parseSystem(systemXml);
  const force = system.forces.find(
    (f): f is PositionRestraintForce => f.type === "PositionRestraint",
  );
  if (force === undefined) {
    throw new Error(
      "the restrained system carries no PositionRestraint; build it with " +
        "tools/build_restraint_fixture.py",
    );
  }

  const measured: RestraintCase[] = [];
  for (const item of cases) {
    const device_ = await evaluateRestraint(
      device,
      force,
      item.positions,
      system.boxVectors,
      system.particleCount,
    );
    const reference = new Float64Array(system.particleCount * 3);
    positionRestraint(force, item.positions, system.boxVectors, reference);

    let worstForceError = 0;
    let worstForceVsReference = 0;
    let openmmMaxForce = 0;
    for (let i = 0; i < reference.length; i++) {
      const expected = item.forces[i] as number;
      const got = device_.forces[i] as number;
      worstForceError = Math.max(worstForceError, Math.abs(got - expected));
      worstForceVsReference = Math.max(
        worstForceVsReference,
        Math.abs(got - (reference[i] as number)),
      );
      openmmMaxForce = Math.max(openmmMaxForce, Math.abs(expected));
    }
    const energyError = Math.abs(device_.energy - item.energy);
    const within = (error: number, floor: number, scale: number): boolean =>
      error < floor || error < RESTRAINT_THRESHOLDS.relative * Math.abs(scale);
    measured.push({
      label: item.label,
      deviceEnergy: device_.energy,
      openmmEnergy: item.energy,
      energyError,
      worstForceError,
      worstForceVsReference,
      openmmMaxForce,
      ok:
        within(energyError, RESTRAINT_THRESHOLDS.energy, item.energy) &&
        within(worstForceError, RESTRAINT_THRESHOLDS.force, openmmMaxForce),
    });
  }

  const assembled = await compareAssembledPotential(
    device,
    systemXml,
    stateXml,
    force,
    0.1,
    50,
  );
  return {
    restrainedAtoms: force.count,
    cases: measured,
    assembled,
    thresholds: RESTRAINT_THRESHOLDS,
    ok: measured.every((item) => item.ok) && assembled.ok,
  };
}
