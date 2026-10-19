/**
 * The acceptance comparison: our kernels against OpenMM's own numbers.
 *
 * Takes the three texts a fixture is made of, so the caller decides how to get
 * them -- `fetch` in a browser, the filesystem in Node -- and the comparison
 * itself is identical in both.
 *
 * Two independent references are used, because they fail differently. The
 * *energy* is compared against `facts.json`, which is what OpenMM's
 * double-precision Reference platform reported for this exact `system.xml`;
 * that catches a wrong formula or a misread parameter. The *forces* are
 * compared against `src/reference/`, our own double-precision implementation
 * evaluated at the same coordinates; that catches a wrong gradient, which an
 * energy comparison cannot see. `state.xml` carries OpenMM's per-atom forces
 * too, but as a total over every force in the system, so they are the gate for
 * the complete force set rather than for one term.
 */

import { parseState } from "../engine/state.js";
import { parseSystem, type Force, type NonbondedForce } from "../engine/system.js";
import { ewaldAlpha } from "../engine/units.js";
import {
  harmonicAngle,
  harmonicBond,
  periodicTorsion,
} from "../reference/forces/bonded.js";
import { directSpace } from "../reference/forces/nonbonded.js";
import { evaluateBondedForce, packPositions, unpackFixed } from "./forces.js";
import { evaluateDirectSpace } from "./nonbonded.js";

export interface ForceComparison {
  readonly forceType: string;
  readonly forceGroup: number;
  readonly termCount: number;
  /** kJ/mol */
  readonly deviceEnergy: number;
  readonly referenceEnergy: number;
  readonly openmmEnergy: number | null;
  readonly energyRelativeError: number;
  readonly energyVsOpenmmRelativeError: number | null;
  /** Relative RMS of (device - reference) over all atoms. */
  readonly forceRelativeRms: number;
  /** kJ/mol/nm */
  readonly forceMaxAbsoluteError: number;
  readonly referenceForceRms: number;
  readonly worstAtom: number;
  readonly ok: boolean;
}

export interface ValidationReport {
  readonly fixture: string;
  readonly particleCount: number;
  readonly openmmVersion: string;
  readonly comparisons: readonly ForceComparison[];
  readonly ok: boolean;
  readonly thresholds: { readonly energyRelative: number; readonly forceRelativeRms: number };
}

/**
 * Thresholds, from AGENTS.md.
 *
 * OpenMM's own single-precision OpenCL platform sits at 1.05e-5 relative force
 * error against its Reference platform on an equilibrated 1L2Y, so ~1e-5 is the
 * floor for a legitimate single-precision implementation rather than a target
 * to beat. These are the bonded terms alone, which are far better conditioned
 * than the nonbonded ones, so they are held an order tighter -- and if that
 * ever needs relaxing, it is a reviewed change with a reason, never a
 * convenience.
 */
export const BONDED_THRESHOLDS = {
  energyRelative: 1e-5,
  forceRelativeRms: 1e-5,
} as const;

/**
 * Direct space is held looser than the bonded terms, and for a measured reason
 * rather than a convenient one.
 *
 * Two things cost accuracy here that the bonded terms do not have. The erfc the
 * kernel uses is a rational approximation good to about 1.2e-7 *relative*
 * (test/erf.test.ts measures it), not an exact function. And the terms cancel
 * hard: for the solvated protein-ligand fixture the exclusion corrections come
 * to +1.67e6 kJ/mol against a direct-group total of +1.56e6, so a relative
 * error in the large terms shows up magnified in the sum.
 */
export const DIRECT_SPACE_THRESHOLDS = {
  energyRelative: 1e-4,
  forceRelativeRms: 1e-4,
} as const;

const REFERENCE: Record<
  string,
  (force: never, positions: Float64Array, forces?: Float64Array) => number
> = {
  HarmonicBondForce: harmonicBond as never,
  HarmonicAngleForce: harmonicAngle as never,
  PeriodicTorsionForce: periodicTorsion as never,
};

function openmmEnergyFor(
  energies: Record<string, number> | undefined,
  forceType: string,
): number | null {
  if (energies === undefined) return null;
  const key = Object.keys(energies).find(
    (k) => k.slice(k.indexOf(":") + 1) === forceType,
  );
  return key === undefined ? null : (energies[key] as number);
}

/**
 * Compare every bonded force in a system, on the device, against both
 * references.
 */
export async function validateBonded(
  device: GPUDevice,
  texts: {
    readonly name: string;
    readonly systemXml: string;
    readonly stateXml: string;
    readonly factsJson?: string;
  },
): Promise<ValidationReport> {
  const system = parseSystem(texts.systemXml);
  const state = parseState(texts.stateXml);
  if (state.particleCount !== system.particleCount) {
    throw new Error(
      `state has ${state.particleCount} particles, system has ${system.particleCount}`,
    );
  }
  const facts =
    texts.factsJson === undefined
      ? undefined
      : (JSON.parse(texts.factsJson) as {
          reference_energies_kJ_mol?: Record<string, number>;
        });

  // The reference is evaluated at the coordinates the device actually sees:
  // positions rounded to the fixed-point format. That isolates the kernel's f32
  // arithmetic, which is what is under test, from the 2.3e-10 nm of rounding
  // the format itself introduces.
  const positionWords = packPositions(state.positions);
  const positions = unpackFixed(positionWords, state.positions.length);

  const comparisons: ForceComparison[] = [];
  for (const force of system.forces) {
    const compute = REFERENCE[force.type];
    if (compute === undefined) continue;
    const bonded = force as Extract<Force, { count: number }>;

    const referenceForces = new Float64Array(system.particleCount * 3);
    const referenceEnergy = compute(force as never, positions, referenceForces);
    const device_ = await evaluateBondedForce(
      device,
      force as never,
      positionWords,
      system.particleCount,
    );

    let sumSquaredError = 0;
    let sumSquaredReference = 0;
    let maxAbsoluteError = 0;
    let worstAtom = 0;
    for (let atom = 0; atom < system.particleCount; atom++) {
      let atomError = 0;
      for (let axis = 0; axis < 3; axis++) {
        const i = atom * 3 + axis;
        const want = referenceForces[i] as number;
        const got = device_.forces[i] as number;
        const error = got - want;
        sumSquaredError += error * error;
        sumSquaredReference += want * want;
        atomError += error * error;
      }
      if (atomError > maxAbsoluteError) {
        maxAbsoluteError = atomError;
        worstAtom = atom;
      }
    }
    const openmm = openmmEnergyFor(facts?.reference_energies_kJ_mol, force.type);
    const energyRelativeError =
      Math.abs(device_.energy - referenceEnergy) / Math.abs(referenceEnergy);
    const forceRelativeRms =
      sumSquaredReference === 0 ? 0 : Math.sqrt(sumSquaredError / sumSquaredReference);
    comparisons.push({
      forceType: force.type,
      forceGroup: force.forceGroup,
      termCount: bonded.count,
      deviceEnergy: device_.energy,
      referenceEnergy,
      openmmEnergy: openmm,
      energyRelativeError,
      energyVsOpenmmRelativeError:
        openmm === null ? null : Math.abs(device_.energy - openmm) / Math.abs(openmm),
      forceRelativeRms,
      forceMaxAbsoluteError: Math.sqrt(maxAbsoluteError),
      referenceForceRms: Math.sqrt(sumSquaredReference / system.particleCount),
      worstAtom,
      ok:
        energyRelativeError < BONDED_THRESHOLDS.energyRelative &&
        forceRelativeRms < BONDED_THRESHOLDS.forceRelativeRms &&
        (openmm === null ||
          Math.abs(device_.energy - openmm) / Math.abs(openmm) <
            BONDED_THRESHOLDS.energyRelative),
    });
  }

  // Direct space: the same two references, with the reciprocal-space half of
  // PME left out of both sides. OpenMM reports it in its own force group, so
  // the comparison is against that group alone.
  const nonbonded = system.forces.find(
    (f): f is NonbondedForce => f.type === "NonbondedForce",
  );
  if (nonbonded !== undefined) {
    const alpha =
      nonbonded.alpha !== 0
        ? nonbonded.alpha
        : ewaldAlpha(nonbonded.cutoff, nonbonded.ewaldTolerance);
    const referenceForces = new Float64Array(system.particleCount * 3);
    const reference = directSpace(
      nonbonded,
      positions,
      system.boxVectors,
      alpha,
      referenceForces,
    );
    const device_ = await evaluateDirectSpace(
      device,
      nonbonded,
      positionWords,
      positions,
      system.boxVectors,
      alpha,
    );

    let sumSquaredError = 0;
    let sumSquaredReference = 0;
    let maxAbsoluteError = 0;
    let worstAtom = 0;
    for (let atom = 0; atom < system.particleCount; atom++) {
      let atomError = 0;
      for (let axis = 0; axis < 3; axis++) {
        const i = atom * 3 + axis;
        const error = (device_.forces[i] as number) - (referenceForces[i] as number);
        sumSquaredError += error * error;
        sumSquaredReference += (referenceForces[i] as number) ** 2;
        atomError += error * error;
      }
      if (atomError > maxAbsoluteError) {
        maxAbsoluteError = atomError;
        worstAtom = atom;
      }
    }
    const openmm = openmmEnergyFor(facts?.reference_energies_kJ_mol, "NonbondedForce");
    const energyRelativeError =
      Math.abs(device_.energy - reference.total) / Math.abs(reference.total);
    const forceRelativeRms =
      sumSquaredReference === 0 ? 0 : Math.sqrt(sumSquaredError / sumSquaredReference);
    const energyVsOpenmm =
      openmm === null ? null : Math.abs(device_.energy - openmm) / Math.abs(openmm);
    comparisons.push({
      forceType: "NonbondedForce[direct]",
      forceGroup: nonbonded.forceGroup,
      termCount: nonbonded.exceptionCount,
      deviceEnergy: device_.energy,
      referenceEnergy: reference.total,
      openmmEnergy: openmm,
      energyRelativeError,
      energyVsOpenmmRelativeError: energyVsOpenmm,
      forceRelativeRms,
      forceMaxAbsoluteError: Math.sqrt(maxAbsoluteError),
      referenceForceRms: Math.sqrt(sumSquaredReference / system.particleCount),
      worstAtom,
      ok:
        energyRelativeError < DIRECT_SPACE_THRESHOLDS.energyRelative &&
        forceRelativeRms < DIRECT_SPACE_THRESHOLDS.forceRelativeRms &&
        (energyVsOpenmm === null ||
          energyVsOpenmm < DIRECT_SPACE_THRESHOLDS.energyRelative),
    });
  }

  return {
    fixture: texts.name,
    particleCount: system.particleCount,
    openmmVersion: system.openmmVersion,
    comparisons,
    ok: comparisons.length > 0 && comparisons.every((c) => c.ok),
    thresholds: BONDED_THRESHOLDS,
  };
}
