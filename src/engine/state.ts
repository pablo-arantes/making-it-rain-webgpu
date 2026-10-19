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
  };
}
