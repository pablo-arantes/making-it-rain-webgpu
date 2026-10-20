/**
 * The engine's other input: an OpenMM `State`.
 *
 * Positions, box vectors, and optionally velocities -- which is what the
 * notebook's `.rst` files hold, so a production run can resume from one.  Our
 * fixtures also carry the Reference platform's per-atom forces, which makes
 * them a force oracle and not only an energy one.
 *
 * Coordinates are kept as `Float64Array`, deliberately.  The engine stores
 * positions in fixed point (see docs/PRECISION.md) and converts on upload; a
 * parser that rounded to f32 on the way in would throw away precision the
 * oracle comparison needs.
 */

import { XmlError, XmlScanner } from "./xml.js";
import type { Vec3 } from "./system.js";

export interface OpenMMState {
  readonly openmmVersion: string;
  readonly stepCount: number;
  /** ps */
  readonly time: number;
  readonly boxVectors: readonly [Vec3, Vec3, Vec3];
  readonly particleCount: number;
  /** nm, xyz interleaved */
  readonly positions: Float64Array;
  /** nm/ps, xyz interleaved; undefined when the state carries none. */
  readonly velocities?: Float64Array;
  /** kJ/mol/nm, xyz interleaved; undefined when the state carries none. */
  readonly forces?: Float64Array;
  /**
   * kJ/mol, when the state was serialised with energies.
   *
   * A convenience that makes a fixture self-describing: the potential energy
   * OpenMM reported for these exact coordinates travels with them.
   */
  readonly potentialEnergy?: number;
  readonly kineticEnergy?: number;
}

function readVectorList(
  scanner: XmlScanner,
  container: string,
  element: string,
): Float64Array {
  const values: number[] = [];
  if (scanner.selfClosing) return new Float64Array(0);
  while (scanner.next()) {
    if (scanner.closing) {
      if (scanner.name === container) break;
      continue;
    }
    if (scanner.name === element) {
      values.push(scanner.num("x"), scanner.num("y"), scanner.num("z"));
    }
  }
  return Float64Array.from(values);
}

export function parseState(xml: string): OpenMMState {
  const scanner = new XmlScanner(xml);
  let openmmVersion = "";
  let stepCount = 0;
  let time = 0;
  let boxVectors: [Vec3, Vec3, Vec3] | undefined;
  let positions: Float64Array | undefined;
  let velocities: Float64Array | undefined;
  let forces: Float64Array | undefined;
  let potentialEnergy: number | undefined;
  let kineticEnergy: number | undefined;
  let sawState = false;

  while (scanner.next()) {
    if (scanner.closing) continue;
    const name = scanner.name;
    if (name === "State") {
      sawState = true;
      openmmVersion = scanner.attr("openmmVersion") ?? "";
      stepCount = scanner.numOr("stepCount", 0);
      time = scanner.numOr("time", 0);
      continue;
    }
    if (!sawState) continue;
    switch (name) {
      case "PeriodicBoxVectors": {
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
        boxVectors = vectors as unknown as [Vec3, Vec3, Vec3];
        break;
      }
      case "Positions":
        positions = readVectorList(scanner, "Positions", "Position");
        break;
      case "Velocities":
        velocities = readVectorList(scanner, "Velocities", "Velocity");
        break;
      case "Forces":
        forces = readVectorList(scanner, "Forces", "Force");
        break;
      case "Energies":
        potentialEnergy = scanner.numOr("PotentialEnergy", Number.NaN);
        kineticEnergy = scanner.numOr("KineticEnergy", Number.NaN);
        break;
      default:
        break;
    }
  }

  if (!sawState) throw new XmlError("no <State> element", 0);
  if (positions === undefined) throw new XmlError("no <Positions>", 0);
  if (boxVectors === undefined) throw new XmlError("no <PeriodicBoxVectors>", 0);
  const particleCount = positions.length / 3;
  for (const [label, array] of [
    ["velocities", velocities],
    ["forces", forces],
  ] as const) {
    if (array !== undefined && array.length !== positions.length) {
      throw new XmlError(
        `${label} has ${array.length / 3} entries but positions has ${particleCount}`,
        0,
      );
    }
  }
  return {
    openmmVersion,
    stepCount,
    time,
    boxVectors,
    particleCount,
    positions,
    // `exactOptionalPropertyTypes` means an absent array is an absent key.
    ...(velocities === undefined ? {} : { velocities }),
    ...(forces === undefined ? {} : { forces }),
    ...(potentialEnergy === undefined || Number.isNaN(potentialEnergy)
      ? {}
      : { potentialEnergy }),
    ...(kineticEnergy === undefined || Number.isNaN(kineticEnergy)
      ? {}
      : { kineticEnergy }),
  };
}

/**
 * Serialise a `State` the way OpenMM's `XmlSerializer` does.
 *
 * This is what the protocol writes as its `.rst` after every stage and stride,
 * and what a resumed run reads back -- the notebook's own checkpoint mechanism,
 * which is how a 12-hour Colab session survives being disconnected. It has to
 * be readable by OpenMM, so that a run started here can be continued in Python
 * and the other way round; `tools/check_state.py` holds it to that.
 *
 * Numbers are written with JavaScript's default `toString`, which produces the
 * shortest decimal that reads back to the same double. OpenMM's own writer
 * emits a different shortest form -- it drops the leading zero, writing `.006`
 * for 0.006 -- and neither is more correct: both parse to the same bits, which
 * is the only property that matters here. Trying to match its formatting
 * character for character would be matching a detail of its float printer.
 *
 * 🔴 Velocities are written when present and omitted when not, and that is not
 * cosmetic. A `State` with no velocities means "set them yourself", and a
 * reader that found zeros instead would start a run at absolute zero without
 * saying so.
 */

export interface SerialiseStateOptions {
  /** Written into the root element; OpenMM writes its own version here. */
  readonly openmmVersion?: string;
  /** Include forces, which a checkpoint does not need and a fixture may want. */
  readonly includeForces?: boolean;
}

const vectorElement = (name: string, x: number, y: number, z: number): string =>
  `\t\t<${name} x="${x}" y="${y}" z="${z}"/>\n`;

function vectorList(
  container: string,
  element: string,
  values: Float64Array,
): string {
  let out = `\t<${container}>\n`;
  for (let i = 0; i < values.length; i += 3) {
    out += vectorElement(
      element,
      values[i] as number,
      values[i + 1] as number,
      values[i + 2] as number,
    );
  }
  return `${out}\t</${container}>\n`;
}

export function serialiseState(
  state: OpenMMState,
  options: SerialiseStateOptions = {},
): string {
  const version = options.openmmVersion ?? state.openmmVersion ?? "8.4";
  let xml = '<?xml version="1.0" ?>\n';
  xml +=
    `<State openmmVersion="${version}" stepCount="${state.stepCount}" ` +
    `time="${state.time}" type="State" version="1">\n`;
  xml += "\t<PeriodicBoxVectors>\n";
  for (const [name, vector] of [
    ["A", state.boxVectors[0]],
    ["B", state.boxVectors[1]],
    ["C", state.boxVectors[2]],
  ] as const) {
    xml += vectorElement(
      name,
      vector[0] as number,
      vector[1] as number,
      vector[2] as number,
    );
  }
  xml += "\t</PeriodicBoxVectors>\n";
  xml += vectorList("Positions", "Position", state.positions);
  if (state.velocities !== undefined) {
    xml += vectorList("Velocities", "Velocity", state.velocities);
  }
  if (options.includeForces === true && state.forces !== undefined) {
    xml += vectorList("Forces", "Force", state.forces);
  }
  xml += "</State>\n";
  return xml;
}
