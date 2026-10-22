/**
 * The triclinic minimum image, against OpenMM and against brute force.
 *
 * The fixture is committed (`tools/dump_triclinic.py` regenerates it) and holds
 * both numbers per pair: the separation OpenMM computed, read out of a
 * `CustomNonbondedForce("r")` so nothing is inferred from a force field, and
 * the minimum over 7³ lattice images. They agree to about 1e-16, so either can
 * be the oracle and the pair of them cross-check each other.
 *
 * 🔴 The test that matters most here is `the reduction is exact below the bound
 * and breaks above it`. `minimumImage` is three subtractions with no search,
 * and that is only correct because the cutoff is bounded by half the narrowest
 * perpendicular width. Nothing in the function's shape says so, and every
 * cubic fixture in this project passes at any cutoff — so without a test that
 * demonstrates the failure *above* the bound, the bound reads like caution and
 * gets widened by whoever next wants a longer cutoff.
 *
 * This file's first version asserted the opposite and was wrong: it concluded
 * a 27-image search was needed, having measured the reduction's error over all
 * separations rather than over the ones inside a cutoff.
 */

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  boxVectorsForShape,
  boxVolume,
  CutoffTooLargeForBox,
  isReducedForm,
  maximumCutoff,
  MAXIMUM_CUTOFF_FRACTION,
  minimumImage,
  nearestImageBySearch,
  NotReducedForm,
  perpendicularWidths,
  reduceBox,
  type BoxVectors,
} from "../src/reference/triclinic.js";
import type { Vec3 } from "../src/engine/system.js";

interface Pair {
  readonly separation: readonly number[];
  readonly openmm: number;
  readonly brute: number;
}
interface Case {
  readonly shape: "cube" | "dodecahedron" | "octahedron";
  readonly width_nm: number;
  readonly requested_vectors: readonly number[][];
  readonly reduced_vectors: readonly number[][];
  readonly already_reduced: boolean;
  readonly volume_nm3: number;
  readonly volume_vs_cube: number;
  readonly pairs: readonly Pair[];
}

const fixture = JSON.parse(
  readFileSync("test/fixtures/triclinic-openmm.json", "utf8"),
) as { readonly openmm_version: string; readonly cases: readonly Case[] };

const asBox = (rows: readonly number[][]): BoxVectors =>
  rows.map((row) => [row[0], row[1], row[2]] as Vec3) as unknown as BoxVectors;

const length = (v: Vec3): number =>
  Math.hypot(v[0] as number, v[1] as number, v[2] as number);

describe("the triclinic minimum image", () => {
  it("reads a fixture with all three of the notebook's shapes", () => {
    expect(fixture.cases.map((c) => c.shape)).toEqual([
      "cube",
      "dodecahedron",
      "octahedron",
    ]);
    expect(fixture.openmm_version).toBeTruthy();
  });

  for (const testCase of fixture.cases) {
    const box = asBox(testCase.reduced_vectors);

    describe(testCase.shape, () => {
      it("builds the vectors OpenMM was asked for", () => {
        const built = boxVectorsForShape(testCase.shape, testCase.width_nm);
        for (let row = 0; row < 3; row++) {
          for (let k = 0; k < 3; k++) {
            expect((built[row] as Vec3)[k] as number).toBeCloseTo(
              testCase.requested_vectors[row]?.[k] as number,
              12,
            );
          }
        }
      });

      it("is already in reduced form, as OpenMM reports", () => {
        // All three shapes come out of `addSolvent` reduced, so `reduceBox` is
        // for a user's own box vectors rather than for these.
        expect(testCase.already_reduced).toBe(true);
        expect(isReducedForm(box)).toBe(true);
        const reducedAgain = reduceBox(box);
        for (let row = 0; row < 3; row++) {
          for (let k = 0; k < 3; k++) {
            expect((reducedAgain[row] as Vec3)[k] as number).toBeCloseTo(
              (box[row] as Vec3)[k] as number,
              12,
            );
          }
        }
      });

      it("agrees with OpenMM on every pair inside the cutoff", () => {
        let worst = 0;
        for (const pair of testCase.pairs) {
          const separation = pair.separation as unknown as Vec3;
          const image = minimumImage([0, 0, 0], separation, box);
          worst = Math.max(worst, Math.abs(length(image) - pair.openmm));
        }
        expect(testCase.pairs.length).toBeGreaterThan(50);
        // f64 round-off only: the two implementations do the same arithmetic in
        // a different order, so there is no approximation to budget for.
        expect(worst).toBeLessThan(1e-12);
      });

      it("agrees with a brute-force minimum over the lattice", () => {
        let worst = 0;
        for (const pair of testCase.pairs) {
          const image = minimumImage([0, 0, 0], pair.separation as unknown as Vec3, box);
          worst = Math.max(worst, Math.abs(length(image) - pair.brute));
        }
        expect(worst).toBeLessThan(1e-12);
      });

      it("reports the volume that makes a non-cubic box worth having", () => {
        expect(boxVolume(box)).toBeCloseTo(testCase.volume_nm3, 9);
        expect(boxVolume(box) / testCase.width_nm ** 3).toBeCloseTo(
          testCase.volume_vs_cube,
          9,
        );
      });

      it("bounds the cutoff by the perpendicular width, not the edge", () => {
        const widths = perpendicularWidths(box);
        const edges = [box[0], box[1], box[2]].map((v) => length(v as Vec3));
        for (let axis = 0; axis < 3; axis++) {
          // A face can never be further away than the edge crossing it is long.
          expect(widths[axis] as number).toBeLessThanOrEqual((edges[axis] as number) + 1e-12);
          expect(widths[axis] as number).toBeGreaterThan(0);
        }
        if (testCase.shape === "cube") {
          for (const width of widths) {
            expect(width as number).toBeCloseTo(testCase.width_nm, 9);
          }
        } else {
          // The whole point: a skewed cell is narrower than it looks.
          expect(Math.min(...(widths as unknown as number[]))).toBeLessThan(
            testCase.width_nm - 1e-9,
          );
        }
      });
    });
  }

  it("🔴 the reduction is exact below the bound and breaks above it", () => {
    // Below the bound, for the pairs OpenMM reported inside a 1.0 nm cutoff,
    // the three-subtraction reduction is the true nearest image everywhere.
    for (const testCase of fixture.cases) {
      const box = asBox(testCase.reduced_vectors);
      expect(maximumCutoff(box)).toBeGreaterThan(1.0);
      let worst = 0;
      for (const pair of testCase.pairs) {
        const image = minimumImage([0, 0, 0], pair.separation as unknown as Vec3, box);
        worst = Math.max(worst, Math.abs(length(image) - pair.brute));
      }
      expect(worst, `${testCase.shape} inside the cutoff`).toBeLessThan(1e-12);
    }

    // And above it, it is wrong -- which is what makes the bound load-bearing.
    // A cube is the exception: its reduction is exact at any separation, which
    // is exactly why the engine's rectangular path can do this and no more.
    const beyond: Record<string, number> = {};
    for (const testCase of fixture.cases) {
      const box = asBox(testCase.reduced_vectors);
      const limit = maximumCutoff(box);
      let wrong = 0;
      let counted = 0;
      // A deterministic sweep rather than random, so the count is stable.
      for (let i = 0; i < 4000; i++) {
        const t = i / 4000;
        const separation: Vec3 = [
          (t * 13.0) % 6.0 - 3.0,
          (t * 17.0) % 6.0 - 3.0,
          (t * 23.0) % 6.0 - 3.0,
        ];
        const exact = nearestImageBySearch([0, 0, 0], separation, box);
        const exactLength = length(exact);
        // Between the bound and 1.3x it, where the measurement found failures.
        if (exactLength < limit || exactLength > 1.3 * limit) continue;
        counted++;
        if (Math.abs(length(minimumImage([0, 0, 0], separation, box)) - exactLength) > 1e-12) {
          wrong++;
        }
      }
      expect(counted, `${testCase.shape} sampled beyond the bound`).toBeGreaterThan(50);
      beyond[testCase.shape] = wrong / counted;
    }
    expect(beyond["cube"] as number).toBe(0);
    expect(beyond["dodecahedron"] as number).toBeGreaterThan(0);
    expect(beyond["octahedron"] as number).toBeGreaterThan(0);
  });

  it("names the bound rather than letting a long cutoff lose pairs", () => {
    const box = boxVectorsForShape("dodecahedron", 3);
    const limit = maximumCutoff(box);
    expect(MAXIMUM_CUTOFF_FRACTION).toBe(0.5);
    // The edge lengths would allow more, which is the trap: an edge-based
    // bound passes and loses pairs.
    expect(limit).toBeLessThan(1.5);
    expect(limit).toBeGreaterThan(1.0);
    const error = new CutoffTooLargeForBox(box, 1.2);
    expect(error.message).toContain("perpendicular width");
    expect(error.message).toContain("1.06");
  });

  it("searches far enough to be an oracle, whatever the separation", () => {
    for (const testCase of fixture.cases) {
      const box = asBox(testCase.reduced_vectors);
      // Separations well beyond the box, which is where an oracle that searched
      // around the raw difference would quietly return the wrong image.
      for (let i = 0; i < 500; i++) {
        const t = i / 500;
        const separation: Vec3 = [
          (t * 29.0) % 24.0 - 12.0,
          (t * 31.0) % 24.0 - 12.0,
          (t * 37.0) % 24.0 - 12.0,
        ];
        // Span 1 and span 2 agree, which is what says 1 is enough.
        expect(length(nearestImageBySearch([0, 0, 0], separation, box, 1))).toBeCloseTo(
          length(nearestImageBySearch([0, 0, 0], separation, box, 2)),
          12,
        );
      }
      // And it agrees with OpenMM on the pairs OpenMM reported.
      for (const pair of testCase.pairs) {
        expect(
          length(nearestImageBySearch([0, 0, 0], pair.separation as unknown as Vec3, box)),
        ).toBeCloseTo(pair.openmm, 12);
      }
    }
  });

  it("refuses a box that is not in reduced form rather than guessing", () => {
    const skewed: BoxVectors = [[3, 0, 0], [2.9, 3, 0], [0, 0, 3]];
    expect(isReducedForm(skewed)).toBe(false);
    expect(() => minimumImage([0, 0, 0], [1, 1, 1], skewed)).toThrow(NotReducedForm);
    // And reducing it makes it acceptable without changing the lattice.
    const fixed = reduceBox(skewed);
    expect(isReducedForm(fixed)).toBe(true);
    expect(boxVolume(fixed)).toBeCloseTo(boxVolume(skewed), 9);
  });
});
