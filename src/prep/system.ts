/**
 * Turning assigned parameters into the `System` the engine runs.
 *
 * Everything else in this directory produces parameters; this produces the
 * object the rest of the codebase already knows how to execute. That matters
 * more than it sounds: the engine contract is OpenMM's `System`, chosen so the
 * notebook stays the oracle and so a user can upload a system prepared
 * elsewhere. Preparation in the browser has to arrive at the *same* object, not
 * at a parallel representation that happens to give similar energies.
 *
 * Which also makes this the strongest available test of the whole preparation
 * path: build a `System` here, parse OpenMM's serialised `System` for the same
 * structure with `parseSystem`, and compare them field by field. That compares
 * against the file the engine consumes rather than against a summary of it.
 *
 * The system *options* -- cutoff, constraints, box -- are deliberately
 * parameters here rather than defaults. They change what is in the `System`
 * without changing any assignment, and conflating the two is what makes a
 * parameter bug look like a settings bug.
 */

import {
  NonbondedMethod,
  type Force,
  type NonbondedMethodValue,
  type OpenMMSystem,
  type Vec3,
} from "../engine/system.js";
import type { PreparedSystem } from "./prepare.js";

export interface SystemOptions {
  readonly nonbondedMethod?: NonbondedMethodValue;
  /** nm; ignored when the method takes no cutoff. */
  readonly cutoff?: number;
  readonly ewaldTolerance?: number;
  readonly boxVectors?: readonly [Vec3, Vec3, Vec3];
  /**
   * Which bonds to replace with constraints. `none` is what a parameter
   * comparison wants; `h-bonds` is what the notebook runs.
   */
  readonly constraints?: "none" | "h-bonds" | "all-bonds";
  readonly removeCMMotion?: boolean;
  /** Reported in the serialised form; only its presence matters to the engine. */
  readonly openmmVersion?: string;
}

const UNIT_BOX: readonly [Vec3, Vec3, Vec3] = [
  [2, 0, 0],
  [0, 2, 0],
  [0, 0, 2],
];

/** Hydrogen by mass, the test OpenMM's `HBonds` applies. */
function isHydrogen(mass: number): boolean {
  return mass < 1.5;
}

export function buildSystem(
  prepared: PreparedSystem,
  options: SystemOptions = {},
): OpenMMSystem {
  const {
    nonbondedMethod = NonbondedMethod.NoCutoff,
    cutoff = 1,
    ewaldTolerance = 5e-4,
    boxVectors = prepared.topology.box ?? UNIT_BOX,
    constraints = "none",
    removeCMMotion = false,
    openmmVersion = "8.4",
  } = options;

  const atoms = prepared.assignment.atoms;
  const count = atoms.length;
  // --- Constraints, which remove the matching bonded terms ----------------
  //
  // 🔴 A constrained bond is not simply also a constraint: OpenMM *drops* the
  // harmonic term, because keeping both would double-count the restoring
  // force. `flexibleConstraints` is the option that keeps it, and it is off by
  // default, so this has to drop them too.
  const constrained: number[] = [];
  const keptBonds: number[] = [];
  prepared.bondTerms.forEach((term, index) => {
    const [a, b] = term.atoms as [number, number];
    const wanted =
      constraints === "all-bonds" ||
      (constraints === "h-bonds" &&
        (isHydrogen((atoms[a] as { mass: number }).mass) ||
          isHydrogen((atoms[b] as { mass: number }).mass)));
    if (wanted) constrained.push(index);
    else keptBonds.push(index);
  });

  const forces: Force[] = [];

  forces.push({
    type: "HarmonicBondForce",
    forceGroup: 0,
    name: "HarmonicBondForce",
    usesPeriodic: false,
    count: keptBonds.length,
    p1: Int32Array.from(keptBonds, (i) => prepared.bondTerms[i]!.atoms[0] as number),
    p2: Int32Array.from(keptBonds, (i) => prepared.bondTerms[i]!.atoms[1] as number),
    length: Float64Array.from(keptBonds, (i) => prepared.bondTerms[i]!.value),
    k: Float64Array.from(keptBonds, (i) => prepared.bondTerms[i]!.k),
  });

  forces.push({
    type: "HarmonicAngleForce",
    forceGroup: 0,
    name: "HarmonicAngleForce",
    usesPeriodic: false,
    count: prepared.angleTerms.length,
    p1: Int32Array.from(prepared.angleTerms, (t) => t.atoms[0] as number),
    p2: Int32Array.from(prepared.angleTerms, (t) => t.atoms[1] as number),
    p3: Int32Array.from(prepared.angleTerms, (t) => t.atoms[2] as number),
    angle: Float64Array.from(prepared.angleTerms, (t) => t.value),
    k: Float64Array.from(prepared.angleTerms, (t) => t.k),
  });

  forces.push({
    type: "PeriodicTorsionForce",
    forceGroup: 0,
    name: "PeriodicTorsionForce",
    usesPeriodic: false,
    count: prepared.torsionTerms.length,
    p1: Int32Array.from(prepared.torsionTerms, (t) => t.atoms[0]),
    p2: Int32Array.from(prepared.torsionTerms, (t) => t.atoms[1]),
    p3: Int32Array.from(prepared.torsionTerms, (t) => t.atoms[2]),
    p4: Int32Array.from(prepared.torsionTerms, (t) => t.atoms[3]),
    periodicity: Int32Array.from(prepared.torsionTerms, (t) => t.periodicity),
    phase: Float64Array.from(prepared.torsionTerms, (t) => t.phase),
    k: Float64Array.from(prepared.torsionTerms, (t) => t.k),
  });

  forces.push({
    type: "NonbondedForce",
    forceGroup: 0,
    name: "NonbondedForce",
    method: nonbondedMethod,
    cutoff,
    ewaldTolerance,
    // 0 means "derive from the tolerance", which is what the notebook leaves it
    // as; the engine reads it the same way.
    alpha: 0,
    grid: [0, 0, 0],
    useDispersionCorrection: true,
    useSwitchingFunction: false,
    switchingDistance: -1,
    exceptionsUsePeriodic: false,
    includeDirectSpace: true,
    recipForceGroup: -1,
    particleCount: count,
    charge: Float64Array.from(atoms, (a) => a.charge),
    sigma: Float64Array.from(atoms, (a) => a.sigma),
    epsilon: Float64Array.from(atoms, (a) => a.epsilon),
    exceptionCount: prepared.exceptions.length,
    exceptionP1: Int32Array.from(prepared.exceptions, (e) => e.atoms[0]),
    exceptionP2: Int32Array.from(prepared.exceptions, (e) => e.atoms[1]),
    exceptionChargeProd: Float64Array.from(prepared.exceptions, (e) => e.chargeProd),
    exceptionSigma: Float64Array.from(prepared.exceptions, (e) => e.sigma),
    exceptionEpsilon: Float64Array.from(prepared.exceptions, (e) => e.epsilon),
  });

  if (removeCMMotion) {
    forces.push({
      type: "CMMotionRemover",
      forceGroup: 0,
      name: "CMMotionRemover",
      frequency: 1,
    });
  }

  return {
    openmmVersion,
    // 🔴 The box comes from the topology whenever it has one, regardless of the
    // nonbonded method. `createSystem` calls `setDefaultPeriodicBoxVectors`
    // whenever the topology carries vectors and only *refuses* when a periodic
    // method is asked for without them, so a solvated system built with
    // NoCutoff still serialises its real box. Tying the box to periodicity
    // instead leaves a 2 nm cube in a file whose contents say otherwise.
    boxVectors,
    particleCount: count,
    masses: Float64Array.from(atoms, (a) => a.mass),
    constraintCount: constrained.length,
    constraintP1: Int32Array.from(constrained, (i) => prepared.bondTerms[i]!.atoms[0] as number),
    constraintP2: Int32Array.from(constrained, (i) => prepared.bondTerms[i]!.atoms[1] as number),
    constraintDistance: Float64Array.from(constrained, (i) => prepared.bondTerms[i]!.value),
    forces,
  };
}
