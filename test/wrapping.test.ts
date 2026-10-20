/**
 * The broken-molecule guard.
 *
 * The engine's bonded forces use plain differences, with no minimum image,
 * because that is what OpenMM does. It is correct while a molecule is never
 * split across the boundary -- which holds during dynamics, since nothing here
 * wraps a coordinate -- and it can fail on *input*, from a state that some
 * other tool wrapped atom by atom.
 *
 * Measured on the device: splitting one molecule costs a factor of 60 in total
 * energy on 1L2Y and 30 on 3HTB (src/gpu/validate-pbc.ts). So this is a guard
 * worth having at the boundary rather than a tidiness check.
 */

import { describe, expect, it } from "vitest";
import {
  BrokenMolecules,
  checkMoleculesIntact,
} from "../src/engine/wrapping.js";
import type { OpenMMSystem, Vec3 } from "../src/engine/system.js";
import { rectangularBox } from "../src/reference/forces/nonbonded.js";
import { fixture, FIXTURE_NAMES } from "./fixtures.js";

const box = [3, 4, 5] as unknown as Vec3;

/** Two particles joined by a constraint, and nothing else. */
const pair = (distance: number): OpenMMSystem =>
  ({
    particleCount: 2,
    masses: Float64Array.from([1, 1]),
    constraintCount: 1,
    constraintP1: Int32Array.from([0]),
    constraintP2: Int32Array.from([1]),
    constraintDistance: Float64Array.from([distance]),
    forces: [],
    boxVectors: [
      [3, 0, 0],
      [0, 4, 0],
      [0, 0, 5],
    ],
    openmmVersion: "8.4",
  }) as unknown as OpenMMSystem;

describe("checkMoleculesIntact", () => {
  it("passes a pair that is close together", () => {
    const report = checkMoleculesIntact(
      pair(0.1),
      Float64Array.from([1, 1, 1, 1.1, 1, 1]),
      box,
    );
    expect(report.intact).toBe(true);
    expect(report.worstSeparation).toBeCloseTo(0.1, 12);
    // Half the *smallest* edge, because that is where the minimum image starts
    // disagreeing with the plain difference on the tightest axis.
    expect(report.limit).toBe(1.5);
  });

  it("catches a pair split across the boundary", () => {
    // One atom at 0.05, the other at 2.95 in a 3 nm box: 2.9 nm apart as the
    // bonded term sees it, 0.1 nm as the nearest image.
    const report = checkMoleculesIntact(
      pair(0.1),
      Float64Array.from([0.05, 1, 1, 2.95, 1, 1]),
      box,
    );
    expect(report.intact).toBe(false);
    expect(report.broken).toHaveLength(1);
    expect(report.broken[0]!.separation).toBeCloseTo(2.9, 12);
    expect(report.broken[0]!.minimumImage).toBeCloseTo(0.1, 12);
    expect(report.broken[0]!.kind).toBe("constraint");
  });

  it("explains itself when thrown", () => {
    const report = checkMoleculesIntact(
      pair(0.1),
      Float64Array.from([0.05, 1, 1, 2.95, 1, 1]),
      box,
    );
    const error = new BrokenMolecules(report);
    expect(error.message).toContain("2.900 nm apart");
    expect(error.message).toContain("0.100 nm for its nearest image");
    // And points at the cause rather than the symptom.
    expect(error.message).toContain("atom by atom");
    expect(error.message).toContain("enforcePeriodicBox=True");
  });

  it("checks harmonic bonds as well as constraints", () => {
    const system = {
      ...pair(0.1),
      constraintCount: 0,
      constraintP1: new Int32Array(0),
      constraintP2: new Int32Array(0),
      constraintDistance: new Float64Array(0),
      forces: [
        {
          type: "HarmonicBondForce",
          count: 1,
          p1: Int32Array.from([0]),
          p2: Int32Array.from([1]),
        },
      ],
    } as unknown as OpenMMSystem;
    const report = checkMoleculesIntact(
      system,
      Float64Array.from([0.05, 1, 1, 2.95, 1, 1]),
      box,
    );
    expect(report.broken[0]!.kind).toBe("bond");
  });

  for (const name of FIXTURE_NAMES) {
    const f = fixture(name);
    it.skipIf(f === null)(`finds ${name}'s own coordinates intact`, () => {
      const report = checkMoleculesIntact(
        f!.system,
        f!.state.positions,
        rectangularBox(f!.state.boxVectors),
      );
      expect(report.intact).toBe(true);
      // A real bonded separation is a fraction of a nanometre, so the margin to
      // the limit is enormous -- which is why this guard cannot misfire on a
      // sane input.
      expect(report.worstSeparation).toBeLessThan(0.5);
      expect(report.limit).toBeGreaterThan(1.4);
    });
  }
});
