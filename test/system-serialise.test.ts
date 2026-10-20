/**
 * Writing `System` XML, and reading it back.
 *
 * The engine's contract is this file format, and it was read-only until the
 * browser started preparing systems. A prepared system that cannot be written
 * out is a result the user has to take on trust: nothing to keep, nothing to
 * re-run, nothing to check in OpenMM.
 *
 * Two tests, and the second is the one with teeth. A round trip proves the
 * writer and our own reader agree, which would still hold if both were wrong
 * in the same way. So the real check is against a file **OpenMM wrote**: parse
 * it, write it back out, parse that, and require the two parses to be
 * identical. That makes OpenMM's own file the fixed point.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  NonbondedMethod,
  parseSystem,
  serialiseSystem,
  type HarmonicAngleForce,
  type HarmonicBondForce,
  type NonbondedForce,
  type OpenMMSystem,
  type PeriodicTorsionForce,
} from "../src/engine/system.js";

/** Everything a comparison should look at, flattened. */
function flatten(system: OpenMMSystem): unknown {
  return {
    version: system.openmmVersion,
    box: system.boxVectors,
    particles: system.particleCount,
    masses: [...system.masses],
    constraints: [
      system.constraintCount,
      [...system.constraintP1],
      [...system.constraintP2],
      [...system.constraintDistance],
    ],
    forces: system.forces.map((force) => {
      switch (force.type) {
        case "HarmonicBondForce": {
          const f = force as HarmonicBondForce;
          return [f.type, f.count, [...f.p1], [...f.p2], [...f.length], [...f.k]];
        }
        case "HarmonicAngleForce": {
          const f = force as HarmonicAngleForce;
          return [
            f.type, f.count, [...f.p1], [...f.p2], [...f.p3], [...f.angle], [...f.k],
          ];
        }
        case "PeriodicTorsionForce": {
          const f = force as PeriodicTorsionForce;
          return [
            f.type, f.count, [...f.p1], [...f.p2], [...f.p3], [...f.p4],
            [...f.periodicity], [...f.phase], [...f.k],
          ];
        }
        case "NonbondedForce": {
          const f = force as NonbondedForce;
          return [
            f.type, f.method, f.cutoff, f.ewaldTolerance, f.alpha, f.grid,
            f.useDispersionCorrection, f.useSwitchingFunction, f.switchingDistance,
            f.exceptionsUsePeriodic, f.includeDirectSpace, f.recipForceGroup,
            [...f.charge], [...f.sigma], [...f.epsilon], f.exceptionCount,
            [...f.exceptionP1], [...f.exceptionP2], [...f.exceptionChargeProd],
            [...f.exceptionSigma], [...f.exceptionEpsilon],
          ];
        }
        default:
          return [force.type, force.name, force.forceGroup];
      }
    }),
  };
}

const FIXTURES = [
  "fixtures/prep/1L2Y-system.xml",
  "fixtures/prep/3HTB-system.xml",
  "fixtures/prep/1L2Y-solv-system.xml",
  "fixtures/1L2Y/system.xml",
];

describe("serialising a System", () => {
  for (const path of FIXTURES) {
    it.skipIf(!existsSync(path))(`round-trips ${path} through OpenMM's own file`, () => {
      // 🔴 OpenMM's file is the fixed point, not ours. A writer and a reader
      // that are wrong in the same way round-trip perfectly; starting from a
      // file OpenMM wrote is what rules that out.
      const theirs = parseSystem(readFileSync(path, "utf8"));
      const ours = parseSystem(serialiseSystem(theirs));
      expect(flatten(ours)).toEqual(flatten(theirs));
    });
  }

  it("writes the elements OpenMM's reader expects", () => {
    // The names and the per-force `version` attribute are what OpenMM's
    // XmlSerializer dispatches on, so a wrong one is a file it refuses rather
    // than a file it misreads. Checked against the shape of a real file.
    const system: OpenMMSystem = {
      openmmVersion: "8.4",
      boxVectors: [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 3],
      ],
      particleCount: 2,
      masses: Float64Array.from([1.008, 16]),
      constraintCount: 1,
      constraintP1: Int32Array.from([0]),
      constraintP2: Int32Array.from([1]),
      constraintDistance: Float64Array.from([0.09572]),
      forces: [
        {
          type: "NonbondedForce",
          forceGroup: 0,
          name: "NonbondedForce",
          method: NonbondedMethod.PME,
          cutoff: 1,
          ewaldTolerance: 5e-4,
          alpha: 0,
          grid: [0, 0, 0],
          useDispersionCorrection: true,
          useSwitchingFunction: false,
          switchingDistance: -1,
          exceptionsUsePeriodic: false,
          includeDirectSpace: true,
          recipForceGroup: -1,
          particleCount: 2,
          charge: Float64Array.from([0.417, -0.834]),
          sigma: Float64Array.from([1, 0.315]),
          epsilon: Float64Array.from([0, 0.636]),
          exceptionCount: 1,
          exceptionP1: Int32Array.from([0]),
          exceptionP2: Int32Array.from([1]),
          exceptionChargeProd: Float64Array.from([0]),
          exceptionSigma: Float64Array.from([1]),
          exceptionEpsilon: Float64Array.from([0]),
        },
      ],
    };
    const xml = serialiseSystem(system);
    expect(xml).toContain('<System openmmVersion="8.4" type="System" version="1">');
    expect(xml).toContain('type="NonbondedForce"');
    expect(xml).toContain('version="4"');
    expect(xml).toContain('<Constraint d="0.09572" p1="0" p2="1"/>');
    expect(xml).toContain("<GlobalParameters/>");
    // And it reads back.
    expect(flatten(parseSystem(xml))).toEqual(flatten(system));
  });
});
