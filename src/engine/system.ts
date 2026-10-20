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

/**
 * Write an `OpenMMSystem` back out as OpenMM `System` XML.
 *
 * The engine's contract is this file format, and until now it was read-only:
 * a system could be loaded but never produced. That is fine while every system
 * comes from the notebook, and not fine once the browser prepares one, because
 * the user then has no artefact — nothing to keep, nothing to re-run, and
 * nothing to check against OpenMM. A prepared system that cannot be serialised
 * is a result you have to take on trust.
 *
 * Written to be read by OpenMM's own `XmlSerializer`, which means the element
 * names, the attribute names and the `version` attribute on each force all
 * have to match what it expects. They are taken from a file OpenMM wrote.
 *
 * Numbers are written with `toString()`, so `0.1522` comes out as `0.1522`
 * where OpenMM writes `.1522`. Both parse to the same double; this is not a
 * byte-for-byte reproduction of OpenMM's writer and does not try to be. What
 * `test/system-serialise.test.ts` does assert is that parsing our output gives
 * back exactly the system we wrote.
 */
export function serialiseSystem(system: OpenMMSystem): string {
  const lines: string[] = [];
  const escape = (value: string): string =>
    value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;");
  const n = (value: number): string => String(value);

  lines.push('<?xml version="1.0" ?>');
  lines.push(
    `<System openmmVersion="${escape(system.openmmVersion)}" type="System" version="1">`,
  );

  lines.push("\t<PeriodicBoxVectors>");
  for (const [index, label] of ["A", "B", "C"].entries()) {
    const vector = system.boxVectors[index] as Vec3;
    lines.push(
      `\t\t<${label} x="${n(vector[0])}" y="${n(vector[1])}" z="${n(vector[2])}"/>`,
    );
  }
  lines.push("\t</PeriodicBoxVectors>");

  lines.push("\t<Particles>");
  for (const mass of system.masses) lines.push(`\t\t<Particle mass="${n(mass)}"/>`);
  lines.push("\t</Particles>");

  if (system.constraintCount === 0) {
    lines.push("\t<Constraints/>");
  } else {
    lines.push("\t<Constraints>");
    for (let i = 0; i < system.constraintCount; i++) {
      lines.push(
        `\t\t<Constraint d="${n(system.constraintDistance[i] as number)}" ` +
          `p1="${system.constraintP1[i]}" p2="${system.constraintP2[i]}"/>`,
      );
    }
    lines.push("\t</Constraints>");
  }

  lines.push("\t<Forces>");
  for (const force of system.forces) {
    const common =
      `forceGroup="${force.forceGroup}" name="${escape(force.name)}"`;
    switch (force.type) {
      case "HarmonicBondForce":
        lines.push(
          `\t\t<Force ${common} type="HarmonicBondForce" ` +
            `usesPeriodic="${force.usesPeriodic ? 1 : 0}" version="2">`,
        );
        lines.push("\t\t\t<Bonds>");
        for (let i = 0; i < force.count; i++) {
          lines.push(
            `\t\t\t\t<Bond d="${n(force.length[i] as number)}" ` +
              `k="${n(force.k[i] as number)}" p1="${force.p1[i]}" p2="${force.p2[i]}"/>`,
          );
        }
        lines.push("\t\t\t</Bonds>");
        lines.push("\t\t</Force>");
        break;

      case "HarmonicAngleForce":
        lines.push(
          `\t\t<Force ${common} type="HarmonicAngleForce" ` +
            `usesPeriodic="${force.usesPeriodic ? 1 : 0}" version="2">`,
        );
        lines.push("\t\t\t<Angles>");
        for (let i = 0; i < force.count; i++) {
          lines.push(
            `\t\t\t\t<Angle a="${n(force.angle[i] as number)}" ` +
              `k="${n(force.k[i] as number)}" p1="${force.p1[i]}" ` +
              `p2="${force.p2[i]}" p3="${force.p3[i]}"/>`,
          );
        }
        lines.push("\t\t\t</Angles>");
        lines.push("\t\t</Force>");
        break;

      case "PeriodicTorsionForce":
        lines.push(
          `\t\t<Force ${common} type="PeriodicTorsionForce" ` +
            `usesPeriodic="${force.usesPeriodic ? 1 : 0}" version="2">`,
        );
        lines.push("\t\t\t<Torsions>");
        for (let i = 0; i < force.count; i++) {
          lines.push(
            `\t\t\t\t<Torsion k="${n(force.k[i] as number)}" p1="${force.p1[i]}" ` +
              `p2="${force.p2[i]}" p3="${force.p3[i]}" p4="${force.p4[i]}" ` +
              `periodicity="${force.periodicity[i]}" ` +
              `phase="${n(force.phase[i] as number)}"/>`,
          );
        }
        lines.push("\t\t\t</Torsions>");
        lines.push("\t\t</Force>");
        break;

      case "NonbondedForce":
        // `rfDielectric` and the LJPME fields are written because OpenMM's
        // reader expects the full attribute set for version 4, even though
        // nothing here uses them.
        lines.push(
          `\t\t<Force alpha="${n(force.alpha)}" cutoff="${n(force.cutoff)}" ` +
            `dispersionCorrection="${force.useDispersionCorrection ? 1 : 0}" ` +
            `ewaldTolerance="${n(force.ewaldTolerance)}" ` +
            `exceptionsUsePeriodic="${force.exceptionsUsePeriodic ? 1 : 0}" ` +
            `${common} includeDirectSpace="${force.includeDirectSpace ? 1 : 0}" ` +
            `ljAlpha="0" ljnx="0" ljny="0" ljnz="0" method="${force.method}" ` +
            `nx="${force.grid[0]}" ny="${force.grid[1]}" nz="${force.grid[2]}" ` +
            `recipForceGroup="${force.recipForceGroup}" rfDielectric="78.3" ` +
            `switchingDistance="${n(force.switchingDistance)}" type="NonbondedForce" ` +
            `useSwitchingFunction="${force.useSwitchingFunction ? 1 : 0}" version="4">`,
        );
        lines.push("\t\t\t<GlobalParameters/>");
        lines.push("\t\t\t<ParticleOffsets/>");
        lines.push("\t\t\t<ExceptionOffsets/>");
        lines.push("\t\t\t<Particles>");
        for (let i = 0; i < force.particleCount; i++) {
          lines.push(
            `\t\t\t\t<Particle eps="${n(force.epsilon[i] as number)}" ` +
              `q="${n(force.charge[i] as number)}" ` +
              `sig="${n(force.sigma[i] as number)}"/>`,
          );
        }
        lines.push("\t\t\t</Particles>");
        if (force.exceptionCount === 0) {
          lines.push("\t\t\t<Exceptions/>");
        } else {
          lines.push("\t\t\t<Exceptions>");
          for (let i = 0; i < force.exceptionCount; i++) {
            lines.push(
              `\t\t\t\t<Exception eps="${n(force.exceptionEpsilon[i] as number)}" ` +
                `p1="${force.exceptionP1[i]}" p2="${force.exceptionP2[i]}" ` +
                `q="${n(force.exceptionChargeProd[i] as number)}" ` +
                `sig="${n(force.exceptionSigma[i] as number)}"/>`,
            );
          }
          lines.push("\t\t\t</Exceptions>");
        }
        lines.push("\t\t</Force>");
        break;

      case "CMMotionRemover":
        lines.push(
          `\t\t<Force ${common} frequency="${force.frequency}" ` +
            'type="CMMotionRemover" version="1"/>',
        );
        break;

      case "MonteCarloBarostat":
        lines.push(
          `\t\t<Force defaultPressure="${n(force.defaultPressure)}" ` +
            `defaultTemperature="${n(force.defaultTemperature)}" ${common} ` +
            `frequency="${force.frequency}" randomNumberSeed="${force.randomSeed}" ` +
            'type="MonteCarloBarostat" version="1"/>',
        );
        break;

      default:
        // A force we can hold but not write would produce a file that silently
        // lost a term, which is worse than refusing.
        throw new UnsupportedForce(
          (force as { type: string }).type,
          "it cannot be serialised, so writing this system would drop it",
        );
    }
  }
  lines.push("\t</Forces>");
  lines.push("</System>");
  return lines.join("\n") + "\n";
}
