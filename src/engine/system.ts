/**
 * The engine's input: an OpenMM `System`, as `XmlSerializer` writes it.
 *
 * This is the contract in `docs/PLAN.md`.  The engine never knows how a system
 * was built, so the browser's own preparation and the Python notebook produce
 * the same artefact and the notebook stays the oracle.  It also means a user
 * can run a `system.xml` they already have.
 *
 * Only the forces this workflow produces are accepted, and anything else is a
 * named error rather than a silent omission.  Measured on both committed
 * fixtures, the GAFF and the OpenFF route each produce exactly five force
 * types -- `HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`,
 * `NonbondedForce` and `CMMotionRemover` -- with no virtual sites and no
 * switching function.  `MonteCarloBarostat` and the restraint
 * `CustomExternalForce` are added by the run, not by the force field.
 */

import { XmlError, XmlScanner } from "./xml.js";

export type Vec3 = readonly [x: number, y: number, z: number];

/** OpenMM's `NonbondedForce::NonbondedMethod`. */
export const NonbondedMethod = {
  NoCutoff: 0,
  CutoffNonPeriodic: 1,
  CutoffPeriodic: 2,
  Ewald: 3,
  PME: 4,
  LJPME: 5,
} as const;
export type NonbondedMethodValue =
  (typeof NonbondedMethod)[keyof typeof NonbondedMethod];

export interface ForceBase {
  readonly type: string;
  readonly forceGroup: number;
  readonly name: string;
}

/** E = (1/2) k (r - length)^2 */
export interface HarmonicBondForce extends ForceBase {
  readonly type: "HarmonicBondForce";
  readonly usesPeriodic: boolean;
  readonly count: number;
  readonly p1: Int32Array;
  readonly p2: Int32Array;
  /** nm */
  readonly length: Float64Array;
  /** kJ/mol/nm^2 */
  readonly k: Float64Array;
}

/** E = (1/2) k (theta - angle)^2 */
export interface HarmonicAngleForce extends ForceBase {
  readonly type: "HarmonicAngleForce";
  readonly usesPeriodic: boolean;
  readonly count: number;
  readonly p1: Int32Array;
  readonly p2: Int32Array;
  readonly p3: Int32Array;
  /** radians */
  readonly angle: Float64Array;
  /** kJ/mol/rad^2 */
  readonly k: Float64Array;
}

/** E = k (1 + cos(periodicity * theta - phase)) */
export interface PeriodicTorsionForce extends ForceBase {
  readonly type: "PeriodicTorsionForce";
  readonly usesPeriodic: boolean;
  readonly count: number;
  readonly p1: Int32Array;
  readonly p2: Int32Array;
  readonly p3: Int32Array;
  readonly p4: Int32Array;
  readonly periodicity: Int32Array;
  /** radians */
  readonly phase: Float64Array;
  /** kJ/mol */
  readonly k: Float64Array;
}

export interface NonbondedForce extends ForceBase {
  readonly type: "NonbondedForce";
  readonly method: NonbondedMethodValue;
  /** nm */
  readonly cutoff: number;
  readonly ewaldTolerance: number;
  /** 0 means "derive from the tolerance", which is what the notebook does. */
  readonly alpha: number;
  readonly grid: Vec3;
  readonly useDispersionCorrection: boolean;
  readonly useSwitchingFunction: boolean;
  readonly switchingDistance: number;
  readonly exceptionsUsePeriodic: boolean;
  readonly includeDirectSpace: boolean;
  /** Which force group reciprocal space reports into; -1 means "the same one". */
  readonly recipForceGroup: number;
  readonly particleCount: number;
  /** elementary charge */
  readonly charge: Float64Array;
  /** nm */
  readonly sigma: Float64Array;
  /** kJ/mol */
  readonly epsilon: Float64Array;
  readonly exceptionCount: number;
  readonly exceptionP1: Int32Array;
  readonly exceptionP2: Int32Array;
  /** e^2 */
  readonly exceptionChargeProd: Float64Array;
  readonly exceptionSigma: Float64Array;
  readonly exceptionEpsilon: Float64Array;
}

export interface CMMotionRemover extends ForceBase {
  readonly type: "CMMotionRemover";
  readonly frequency: number;
}

export interface MonteCarloBarostat extends ForceBase {
  readonly type: "MonteCarloBarostat";
  /** bar */
  readonly defaultPressure: number;
  /** K */
  readonly defaultTemperature: number;
  readonly frequency: number;
  readonly randomSeed: number;
}

/**
 * The position restraint the notebook's equilibration adds.
 *
 * Only this one expression is accepted -- `k*periodicdistance(x,y,z,x0,y0,z0)^2`
 * with per-particle `k, x0, y0, z0` -- and it is implemented directly rather
 * than through an expression parser, as `docs/PLAN.md` specifies.  A
 * `CustomExternalForce` carrying anything else is an error, because silently
 * evaluating the wrong energy is worse than refusing.
 */
export interface PositionRestraintForce extends ForceBase {
  readonly type: "PositionRestraint";
  readonly energyExpression: string;
  readonly count: number;
  readonly particle: Int32Array;
  /** kJ/mol/nm^2 */
  readonly k: Float64Array;
  /** nm */
  readonly x0: Float64Array;
  readonly y0: Float64Array;
  readonly z0: Float64Array;
}

export type Force =
  | HarmonicBondForce
  | HarmonicAngleForce
  | PeriodicTorsionForce
  | NonbondedForce
  | CMMotionRemover
  | MonteCarloBarostat
  | PositionRestraintForce;

export interface OpenMMSystem {
  readonly openmmVersion: string;
  readonly boxVectors: readonly [Vec3, Vec3, Vec3];
  readonly particleCount: number;
  /** amu */
  readonly masses: Float64Array;
  readonly constraintCount: number;
  readonly constraintP1: Int32Array;
  readonly constraintP2: Int32Array;
  /** nm */
  readonly constraintDistance: Float64Array;
  readonly forces: readonly Force[];
}

/** The restraint expression the engine implements, whitespace removed. */
const RESTRAINT_EXPRESSION = "k*periodicdistance(x,y,z,x0,y0,z0)^2";

export class UnsupportedForce extends Error {
  constructor(readonly forceType: string, detail?: string) {
    super(
      `${forceType} is not implemented by this engine` +
        (detail === undefined ? "" : `: ${detail}`) +
        ". The supported set is HarmonicBondForce, HarmonicAngleForce, " +
        "PeriodicTorsionForce, NonbondedForce, CMMotionRemover, " +
        "MonteCarloBarostat, and a CustomExternalForce holding exactly " +
        `"${RESTRAINT_EXPRESSION}". See docs/PLAN.md.`,
    );
    this.name = "UnsupportedForce";
  }
}

function readBoxVectors(scanner: XmlScanner): [Vec3, Vec3, Vec3] {
  const vectors: Vec3[] = [];
  while (scanner.next()) {
    if (scanner.closing && scanner.name === "PeriodicBoxVectors") break;
    if (scanner.name === "A" || scanner.name === "B" || scanner.name === "C") {
      vectors.push([scanner.num("x"), scanner.num("y"), scanner.num("z")]);
    }
  }
  if (vectors.length !== 3) {
    throw new XmlError(
      `expected three box vectors, found ${vectors.length}`,
      scanner.offset,
    );
  }
  return vectors as unknown as [Vec3, Vec3, Vec3];
}

/** Walk the children of the element the scanner is on, by tag name. */
function eachChild(
  scanner: XmlScanner,
  container: string,
  visit: (name: string) => void,
): void {
  if (scanner.selfClosing) return;
  while (scanner.next()) {
    if (scanner.closing) {
      if (scanner.name === container) return;
      continue;
    }
    visit(scanner.name);
    // Nested containers inside a force (GlobalParameters, ParticleOffsets and
    // the rest) are handled by their own visitor or skipped by it.
  }
  throw new XmlError(`<${container}> is never closed`, scanner.offset);
}

function readHarmonicBondForce(
  scanner: XmlScanner,
  base: ForceBase & { usesPeriodic: boolean },
): HarmonicBondForce {
  const p1: number[] = [];
  const p2: number[] = [];
  const length: number[] = [];
  const k: number[] = [];
  eachChild(scanner, "Force", (name) => {
    if (name === "Bond") {
      p1.push(scanner.int("p1"));
      p2.push(scanner.int("p2"));
      length.push(scanner.num("d"));
      k.push(scanner.num("k"));
    }
  });
  return {
    ...base,
    type: "HarmonicBondForce",
    count: p1.length,
    p1: Int32Array.from(p1),
    p2: Int32Array.from(p2),
    length: Float64Array.from(length),
    k: Float64Array.from(k),
  };
}

function readHarmonicAngleForce(
  scanner: XmlScanner,
  base: ForceBase & { usesPeriodic: boolean },
): HarmonicAngleForce {
  const p1: number[] = [];
  const p2: number[] = [];
  const p3: number[] = [];
  const angle: number[] = [];
  const k: number[] = [];
  eachChild(scanner, "Force", (name) => {
    if (name === "Angle") {
      p1.push(scanner.int("p1"));
      p2.push(scanner.int("p2"));
      p3.push(scanner.int("p3"));
      angle.push(scanner.num("a"));
      k.push(scanner.num("k"));
    }
  });
  return {
    ...base,
    type: "HarmonicAngleForce",
    count: p1.length,
    p1: Int32Array.from(p1),
    p2: Int32Array.from(p2),
    p3: Int32Array.from(p3),
    angle: Float64Array.from(angle),
    k: Float64Array.from(k),
  };
}

function readPeriodicTorsionForce(
  scanner: XmlScanner,
  base: ForceBase & { usesPeriodic: boolean },
): PeriodicTorsionForce {
  const p1: number[] = [];
  const p2: number[] = [];
  const p3: number[] = [];
  const p4: number[] = [];
  const periodicity: number[] = [];
  const phase: number[] = [];
  const k: number[] = [];
  eachChild(scanner, "Force", (name) => {
    if (name === "Torsion") {
      p1.push(scanner.int("p1"));
      p2.push(scanner.int("p2"));
      p3.push(scanner.int("p3"));
      p4.push(scanner.int("p4"));
      periodicity.push(scanner.int("periodicity"));
      phase.push(scanner.num("phase"));
      k.push(scanner.num("k"));
    }
  });
  return {
    ...base,
    type: "PeriodicTorsionForce",
    count: p1.length,
    p1: Int32Array.from(p1),
    p2: Int32Array.from(p2),
    p3: Int32Array.from(p3),
    p4: Int32Array.from(p4),
    periodicity: Int32Array.from(periodicity),
    phase: Float64Array.from(phase),
    k: Float64Array.from(k),
  };
}

function readNonbondedForce(scanner: XmlScanner, base: ForceBase): NonbondedForce {
  const method = scanner.int("method") as NonbondedMethodValue;
  const header = {
    method,
    cutoff: scanner.num("cutoff"),
    ewaldTolerance: scanner.num("ewaldTolerance"),
    alpha: scanner.numOr("alpha", 0),
    grid: [scanner.numOr("nx", 0), scanner.numOr("ny", 0), scanner.numOr("nz", 0)] as Vec3,
    useDispersionCorrection: scanner.flag("dispersionCorrection"),
    useSwitchingFunction: scanner.flag("useSwitchingFunction"),
    switchingDistance: scanner.numOr("switchingDistance", -1),
    exceptionsUsePeriodic: scanner.flag("exceptionsUsePeriodic"),
    includeDirectSpace: scanner.flag("includeDirectSpace", true),
    recipForceGroup: scanner.numOr("recipForceGroup", -1),
  };

  const charge: number[] = [];
  const sigma: number[] = [];
  const epsilon: number[] = [];
  const exceptionP1: number[] = [];
  const exceptionP2: number[] = [];
  const exceptionChargeProd: number[] = [];
  const exceptionSigma: number[] = [];
  const exceptionEpsilon: number[] = [];
  // Parameter offsets make a force depend on a global parameter. Nothing in
  // this workflow creates one, so an offset present would silently change the
  // energy and is refused rather than ignored.
  let offsets = 0;
  eachChild(scanner, "Force", (name) => {
    switch (name) {
      case "Particle":
        charge.push(scanner.num("q"));
        sigma.push(scanner.num("sig"));
        epsilon.push(scanner.num("eps"));
        break;
      case "Exception":
        exceptionP1.push(scanner.int("p1"));
        exceptionP2.push(scanner.int("p2"));
        exceptionChargeProd.push(scanner.num("q"));
        exceptionSigma.push(scanner.num("sig"));
        exceptionEpsilon.push(scanner.num("eps"));
        break;
      case "ParticleOffset":
      case "ExceptionOffset":
        offsets++;
        break;
      default:
        break;
    }
  });
  if (offsets > 0) {
    throw new UnsupportedForce(
      "NonbondedForce",
      `it carries ${offsets} parameter offset(s), which make the energy depend ` +
        "on a global parameter",
    );
  }
  if (method === NonbondedMethod.LJPME) {
    throw new UnsupportedForce("NonbondedForce", "LJ-PME is not implemented");
  }
  return {
    ...base,
    type: "NonbondedForce",
    ...header,
    particleCount: charge.length,
    charge: Float64Array.from(charge),
    sigma: Float64Array.from(sigma),
    epsilon: Float64Array.from(epsilon),
    exceptionCount: exceptionP1.length,
    exceptionP1: Int32Array.from(exceptionP1),
    exceptionP2: Int32Array.from(exceptionP2),
    exceptionChargeProd: Float64Array.from(exceptionChargeProd),
    exceptionSigma: Float64Array.from(exceptionSigma),
    exceptionEpsilon: Float64Array.from(exceptionEpsilon),
  };
}

function readPositionRestraint(
  scanner: XmlScanner,
  base: ForceBase,
): PositionRestraintForce {
  const expression = scanner.str("energy");
  // OpenMM keeps the expression verbatim, including the notebook's trailing
  // semicolon and its spaces.
  const normalised = expression.replace(/[\s;]/g, "");
  if (normalised !== RESTRAINT_EXPRESSION) {
    throw new UnsupportedForce(
      "CustomExternalForce",
      `its energy expression is "${expression}", and this engine implements ` +
        `only "${RESTRAINT_EXPRESSION}"`,
    );
  }
  const parameterNames: string[] = [];
  const globalNames: string[] = [];
  const particle: number[] = [];
  const values: number[][] = [];
  // 🔴 The element names here are OpenMM's, taken from a real serialised force
  // rather than guessed: the parameters are `<Parameter name="k"/>` nested
  // inside `<PerParticleParameters>` -- not `<PerParticleParameter>` -- and a
  // particle's index attribute is `index`, not `p`. The first version of this
  // parser had both wrong and had never met an actual `CustomExternalForce`,
  // because no fixture carries one until a run adds it.
  //
  // `eachChild` reports every descendant's name without tracking depth, so
  // which container a `<Parameter>` belongs to has to be tracked here. It
  // matters: a global `k` and a per-particle `k` are different formulations of
  // the same restraint, and only the second is what the notebook writes.
  let container = "";
  eachChild(scanner, "Force", (name) => {
    if (name === "PerParticleParameters" || name === "GlobalParameters") {
      container = name;
    } else if (name === "Parameter") {
      if (container === "GlobalParameters") globalNames.push(scanner.str("name"));
      else parameterNames.push(scanner.str("name"));
    } else if (name === "Particle") {
      particle.push(scanner.int("index"));
      // Per-particle parameters are written as param1..paramN in declaration
      // order, so the names tell us which column is which.
      values.push(parameterNames.map((_, i) => scanner.num(`param${i + 1}`)));
    }
  });
  if (globalNames.length > 0) {
    throw new UnsupportedForce(
      "CustomExternalForce",
      `it has global parameters (${globalNames.join(", ")}). This engine reads ` +
        "k, x0, y0, z0 per particle, which is what the notebook writes; a " +
        "global parameter would change what the expression means and is not " +
        "silently substituted",
    );
  }
  const column = (want: string): Float64Array => {
    const index = parameterNames.indexOf(want);
    if (index < 0) {
      throw new UnsupportedForce(
        "CustomExternalForce",
        `it has no per-particle parameter "${want}" (found ${parameterNames.join(", ")})`,
      );
    }
    return Float64Array.from(values, (row) => row[index] as number);
  };
  return {
    ...base,
    type: "PositionRestraint",
    energyExpression: expression,
    count: particle.length,
    particle: Int32Array.from(particle),
    k: column("k"),
    x0: column("x0"),
    y0: column("y0"),
    z0: column("z0"),
  };
}

function readForce(scanner: XmlScanner): Force {
  const type = scanner.str("type");
  const base: ForceBase = {
    type,
    forceGroup: scanner.numOr("forceGroup", 0),
    name: scanner.attr("name") ?? type,
  };
  const periodic = { ...base, usesPeriodic: scanner.flag("usesPeriodic") };
  switch (type) {
    case "HarmonicBondForce":
      return readHarmonicBondForce(scanner, periodic);
    case "HarmonicAngleForce":
      return readHarmonicAngleForce(scanner, periodic);
    case "PeriodicTorsionForce":
      return readPeriodicTorsionForce(scanner, periodic);
    case "NonbondedForce":
      return readNonbondedForce(scanner, base);
    case "CMMotionRemover": {
      const frequency = scanner.numOr("frequency", 1);
      scanner.skipElement();
      return { ...base, type: "CMMotionRemover", frequency };
    }
    case "MonteCarloBarostat": {
      const force: MonteCarloBarostat = {
        ...base,
        type: "MonteCarloBarostat",
        defaultPressure: scanner.num("pressure"),
        defaultTemperature: scanner.num("temperature"),
        frequency: scanner.numOr("frequency", 25),
        randomSeed: scanner.numOr("randomSeed", 0),
      };
      scanner.skipElement();
      return force;
    }
    case "CustomExternalForce":
      return readPositionRestraint(scanner, base);
    default:
      throw new UnsupportedForce(type);
  }
}

/**
 * Parse an OpenMM `System` XML document.
 *
 * Measured on the committed fixtures: 712 kB / 2349 particles for 1L2Y and
 * 7.3 MB / 26078 particles for 3HTB.
 */
export function parseSystem(xml: string): OpenMMSystem {
  const scanner = new XmlScanner(xml);
  let openmmVersion = "";
  let boxVectors: [Vec3, Vec3, Vec3] | undefined;
  const masses: number[] = [];
  const constraintP1: number[] = [];
  const constraintP2: number[] = [];
  const constraintDistance: number[] = [];
  const forces: Force[] = [];
  let sawSystem = false;
  let depth = 0;

  while (scanner.next()) {
    if (scanner.closing) {
      depth--;
      continue;
    }
    const name = scanner.name;
    if (name === "System") {
      sawSystem = true;
      openmmVersion = scanner.attr("openmmVersion") ?? "";
      if (!scanner.selfClosing) depth++;
      continue;
    }
    if (!sawSystem) continue;
    switch (name) {
      case "PeriodicBoxVectors":
        boxVectors = readBoxVectors(scanner);
        break;
      case "Particles":
        eachChild(scanner, "Particles", (child) => {
          if (child === "Particle") masses.push(scanner.num("mass"));
        });
        break;
      case "Constraints":
        eachChild(scanner, "Constraints", (child) => {
          if (child === "Constraint") {
            constraintP1.push(scanner.int("p1"));
            constraintP2.push(scanner.int("p2"));
            constraintDistance.push(scanner.num("d"));
          }
        });
        break;
      case "Forces":
        eachChild(scanner, "Forces", (child) => {
          if (child === "Force") forces.push(readForce(scanner));
        });
        break;
      default:
        if (!scanner.selfClosing) depth++;
        break;
    }
  }

  if (!sawSystem) throw new XmlError("no <System> element", 0);
  if (boxVectors === undefined) throw new XmlError("no <PeriodicBoxVectors>", 0);

  const nonbonded = forces.find((f) => f.type === "NonbondedForce");
  if (nonbonded !== undefined && nonbonded.particleCount !== masses.length) {
    throw new XmlError(
      `NonbondedForce has ${nonbonded.particleCount} particles but the System ` +
        `has ${masses.length}`,
      0,
    );
  }
  return {
    openmmVersion,
    boxVectors,
    particleCount: masses.length,
    masses: Float64Array.from(masses),
    constraintCount: constraintP1.length,
    constraintP1: Int32Array.from(constraintP1),
    constraintP2: Int32Array.from(constraintP2),
    constraintDistance: Float64Array.from(constraintDistance),
    forces,
  };
}
