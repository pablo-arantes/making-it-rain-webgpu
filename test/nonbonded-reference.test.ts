import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseState } from "../src/engine/state.js";
import { parseSystem, type NonbondedForce } from "../src/engine/system.js";
import { ewaldAlpha } from "../src/engine/units.js";
import {
  directSpace,
  directSpaceNaive,
  dispersionCorrection,
  dispersionCorrectionNaive,
  rectangularBox,
  selfEnergy,
} from "../src/reference/forces/nonbonded.js";
import { fixture, FIXTURE_NAMES, referenceEnergy } from "./fixtures.js";

/**
 * The direct-space reference against OpenMM's own direct-space force group.
 *
 * The small fixture is committed and hand-built
 * (tools/build_nonbonded_fixture.py), so this runs with no conda environment;
 * the solvated fixtures are the acceptance gate.
 */

interface NonbondedFacts {
  readonly box_nm: number;
  readonly cutoff_nm: number;
  readonly ewald_tolerance: number;
  readonly PME: {
    readonly alpha_per_nm: number;
    readonly grid: number[];
    readonly direct_group: number;
    readonly reciprocal_group: number;
  };
  readonly Ewald: { readonly alpha_per_nm: number; readonly direct_group: number };
  readonly decomposition: Record<string, number>;
}

const small = (() => {
  try {
    return {
      system: parseSystem(readFileSync("test/fixtures/nonbonded-system.xml", "utf8")),
      state: parseState(readFileSync("test/fixtures/nonbonded-state.xml", "utf8")),
      facts: JSON.parse(
        readFileSync("test/fixtures/nonbonded-facts.json", "utf8"),
      ) as NonbondedFacts,
    };
  } catch (error) {
    // Reporting the reason rather than only the absence: "not generated" and
    // "generated but unparseable" are different problems and looked the same
    // here once already.
    process.stderr.write(
      `\n  test/fixtures/nonbonded-*.xml could not be loaded: ` +
        `${String(error).split("\n")[0]}\n` +
        "  Regenerate with: python tools/build_nonbonded_fixture.py --out test/fixtures\n\n",
    );
    return null;
  }
})();

describe("the committed nonbonded fixture", () => {
  it("is present", () => {
    expect(small, "regenerate with tools/build_nonbonded_fixture.py").not.toBeNull();
  });

  const nonbonded = (): NonbondedForce =>
    small!.system.forces.find((f) => f.type === "NonbondedForce") as NonbondedForce;

  it("reproduces OpenMM's direct group, term by term", () => {
    const force = nonbonded();
    const alpha = small!.facts.PME.alpha_per_nm;
    const result = directSpaceNaive(
      force,
      small!.state.positions,
      small!.system.boxVectors,
      alpha,
    );
    const d = small!.facts.decomposition;
    // Each term against the Python decomposition, which itself agrees with
    // OpenMM to 2e-9. Checking them separately means a sign error in one term
    // cannot hide behind another.
    expect(result.coulomb).toBeCloseTo(d.direct_coulomb as number, 9);
    expect(result.lennardJones).toBeCloseTo(d.direct_lj as number, 9);
    expect(result.exclusionCorrection).toBeCloseTo(d.exclusion_correction as number, 8);
    expect(result.exceptions).toBeCloseTo(d.exception_energy as number, 8);
    // And the total against OpenMM itself.
    expect(result.total).toBeCloseTo(small!.facts.PME.direct_group, 6);
  });

  it("reproduces OpenMM's dispersion correction", () => {
    const force = nonbonded();
    const box = rectangularBox(small!.system.boxVectors);
    const volume = box[0] * box[1] * box[2];
    // Measured against OpenMM by toggling setUseDispersionCorrection: it is
    // -0.020466041 kJ/mol for this system.
    expect(dispersionCorrection(force, volume)).toBeCloseTo(-0.020466041, 9);
    // And the grouped implementation against the brute-force one.
    expect(dispersionCorrection(force, volume)).toBeCloseTo(
      dispersionCorrectionNaive(force, volume),
      12,
    );
  });

  it("reproduces the self-energy term", () => {
    const force = nonbonded();
    const alpha = small!.facts.PME.alpha_per_nm;
    expect(selfEnergy(force, alpha)).toBeCloseTo(
      small!.facts.decomposition.self_energy as number,
      8,
    );
  });

  it("derives the same alpha OpenMM used", () => {
    const force = nonbonded();
    expect(ewaldAlpha(force.cutoff, force.ewaldTolerance)).toBeCloseTo(
      small!.facts.PME.alpha_per_nm,
      12,
    );
  });

  it("gives forces that match a numerical derivative", () => {
    const force = nonbonded();
    const alpha = small!.facts.PME.alpha_per_nm;
    const analytic = new Float64Array(small!.system.particleCount * 3);
    directSpaceNaive(
      force,
      small!.state.positions,
      small!.system.boxVectors,
      alpha,
      analytic,
    );
    const positions = Float64Array.from(small!.state.positions);
    const h = 1e-6;
    for (let i = 0; i < positions.length; i++) {
      const original = positions[i] as number;
      positions[i] = original + h;
      const plus = directSpaceNaive(force, positions, small!.system.boxVectors, alpha).total;
      positions[i] = original - h;
      const minus = directSpaceNaive(force, positions, small!.system.boxVectors, alpha).total;
      positions[i] = original;
      const numerical = -(plus - minus) / (2 * h);
      const magnitude = Math.max(Math.abs(numerical), Math.abs(analytic[i] as number), 1);
      expect(Math.abs((analytic[i] as number) - numerical) / magnitude).toBeLessThan(1e-6);
    }
  });

  it("agrees with the cell-list implementation", () => {
    const force = nonbonded();
    const alpha = small!.facts.PME.alpha_per_nm;
    const naive = directSpaceNaive(force, small!.state.positions, small!.system.boxVectors, alpha);
    const cells = directSpace(force, small!.state.positions, small!.system.boxVectors, alpha);
    expect(cells.total).toBeCloseTo(naive.total, 10);
    expect(cells.pairsWithinCutoff).toBe(naive.pairsWithinCutoff);
  });
});

for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  describe.skipIf(f === null)(`direct space for ${name}`, () => {
    const nonbonded = (): NonbondedForce =>
      f!.system.forces.find((x) => x.type === "NonbondedForce") as NonbondedForce;

    it("reproduces OpenMM's direct-space force group", () => {
      const force = nonbonded();
      const alpha = f!.facts.pme.alpha_per_nm;
      const result = directSpace(force, f!.state.positions, f!.system.boxVectors, alpha);
      const want = referenceEnergy(f!, "NonbondedForce");
      // Double against double; the difference is summation order over a few
      // million terms that cancel against each other.
      expect(Math.abs(result.total - want) / Math.abs(want)).toBeLessThan(1e-10);
    });

    it("groups the dispersion correction without changing it", () => {
      const force = nonbonded();
      const box = rectangularBox(f!.system.boxVectors);
      const volume = box[0] * box[1] * box[2];
      const grouped = dispersionCorrection(force, volume);
      expect(Number.isFinite(grouped)).toBe(true);
      if (force.particleCount <= 4000) {
        // Brute force is N(N+1)/2 pairs; 2373 particles is 2.8e6, which is
        // affordable, and 26749 would be 3.6e8, which is not.
        //
        // 1e-10 rather than something tighter, and the grouped version is the
        // more accurate of the two: brute force adds 2.8e6 terms one at a time
        // where grouping adds a few thousand with multiplicities, so the two
        // differ by f64 summation order at the 1e-11 level.
        expect(Math.abs(grouped - dispersionCorrectionNaive(force, volume)) /
          Math.abs(grouped)).toBeLessThan(1e-10);
      }
    });

    it("has an exclusion correction that dominates the direct group", () => {
      // The thing a naive pair sum would get wrong. For a water box the
      // corrections are strongly positive and larger than the total energy.
      const force = nonbonded();
      const alpha = f!.facts.pme.alpha_per_nm;
      const result = directSpace(force, f!.state.positions, f!.system.boxVectors, alpha);
      expect(result.exclusionCorrection).toBeGreaterThan(0);
      expect(result.exclusionCorrection).toBeGreaterThan(Math.abs(result.coulomb));
    });

    it("agrees with the brute-force implementation", () => {
      const force = nonbonded();
      const alpha = f!.facts.pme.alpha_per_nm;
      const cells = directSpace(force, f!.state.positions, f!.system.boxVectors, alpha);
      if (force.particleCount > 4000) {
        // O(N^2) on 26749 particles is 3.6e8 pairs; the cell list is checked
        // against it on the smaller fixture and on the committed one.
        expect(cells.pairsWithinCutoff).toBeGreaterThan(0);
        return;
      }
      const naive = directSpaceNaive(force, f!.state.positions, f!.system.boxVectors, alpha);
      expect(cells.pairsWithinCutoff).toBe(naive.pairsWithinCutoff);
      expect(Math.abs(cells.total - naive.total) / Math.abs(naive.total)).toBeLessThan(1e-12);
    });
  });
}
