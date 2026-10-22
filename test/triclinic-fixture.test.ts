/**
 * OpenMM's own dodecahedral solvation, against our box construction and bound.
 *
 * `tools/build_reference_system.py --box-shape dodecahedron` builds it, and
 * like every fixture it is not committed, so this skips when it is absent.
 *
 * What it is for: the triclinic numerics in `src/reference/triclinic.ts` are
 * tested against a synthetic 3 nm box, and a synthetic box cannot tell whether
 * the construction is the one OpenMM actually produces for a real solute, nor
 * whether the cutoff bound those numerics need is a bound a real system
 * satisfies. Both turn out to be true, and the second only just — which is
 * exactly the kind of margin worth pinning before anyone widens a cutoff.
 */

import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { parseState } from "../src/engine/state.js";
import { parseSystem } from "../src/engine/system.js";
import {
  boxVectorsForShape,
  boxVolume,
  isReducedForm,
  maximumCutoff,
  perpendicularWidths,
  type BoxVectors,
} from "../src/reference/triclinic.js";
import type { NonbondedForce, Vec3 } from "../src/engine/system.js";

const dodecahedron = "fixtures/1L2Y-dodecahedron";
const cube = "fixtures/1L2Y";
const present = existsSync(`${dodecahedron}/state.xml`) && existsSync(`${cube}/facts.json`);

describe("OpenMM's dodecahedral solvation", () => {
  it.skipIf(!present)(
    "is the box our own construction builds, element for element",
    () => {
      const state = parseState(readFileSync(`${dodecahedron}/state.xml`, "utf8"));
      const box = state.boxVectors as unknown as BoxVectors;
      const width = (box[0] as Vec3)[0] as number;
      const built = boxVectorsForShape("dodecahedron", width);
      for (let row = 0; row < 3; row++) {
        for (let k = 0; k < 3; k++) {
          // Exactly, not approximately: both compute d/2 and d*sqrt(2)/2 from
          // the same d in f64, so any difference would be a different formula.
          expect((built[row] as Vec3)[k] as number).toBe((box[row] as Vec3)[k] as number);
        }
      }
      expect(isReducedForm(box)).toBe(true);
    },
  );

  it.skipIf(!present)("holds 70.7% of the enclosing cube, which is the point", () => {
    const state = parseState(readFileSync(`${dodecahedron}/state.xml`, "utf8"));
    const box = state.boxVectors as unknown as BoxVectors;
    const width = (box[0] as Vec3)[0] as number;
    expect(boxVolume(box) / width ** 3).toBeCloseTo(Math.SQRT1_2, 9);

    // And the water count follows the volume, which is the saving a user gets.
    const dodecaFacts = JSON.parse(
      readFileSync(`${dodecahedron}/facts.json`, "utf8"),
    ) as { residue_counts: Record<string, number>; atoms_total: number };
    const cubeFacts = JSON.parse(readFileSync(`${cube}/facts.json`, "utf8")) as {
      residue_counts: Record<string, number>;
      atoms_total: number;
    };
    const ratio =
      (dodecaFacts.residue_counts["HOH"] as number) /
      (cubeFacts.residue_counts["HOH"] as number);
    // Not exactly the volume ratio: the solute is the same size in both, the
    // padding is measured to the solute rather than between images, and whole
    // molecules are kept or dropped. 0.75 against the volume's 0.707.
    expect(ratio).toBeGreaterThan(0.65);
    expect(ratio).toBeLessThan(0.85);
    expect(dodecaFacts.atoms_total).toBeLessThan(cubeFacts.atoms_total);
  });

  it.skipIf(!present)(
    "🔴 satisfies the cutoff bound, and only just — so the bound is live",
    () => {
      const state = parseState(readFileSync(`${dodecahedron}/state.xml`, "utf8"));
      const system = parseSystem(readFileSync(`${dodecahedron}/system.xml`, "utf8"));
      const box = state.boxVectors as unknown as BoxVectors;
      const force = system.forces.find(
        (f): f is NonbondedForce => f.type === "NonbondedForce",
      );
      expect(force).toBeDefined();
      const limit = maximumCutoff(box);
      const cutoff = (force as NonbondedForce).cutoff;

      // OpenMM's own default solvation fits.
      expect(cutoff).toBeLessThan(limit);
      // But by 9%, not by a comfortable margin: 1.0 nm against 1.0888. A user
      // choosing the 1.2 nm cutoff that other protocols use would be outside
      // it, so this has to be checked per system and reported, never assumed.
      expect(limit / cutoff).toBeLessThan(1.15);

      // And an edge-based bound would have waved it through at 1.5, which is
      // the trap: every edge of a rhombic dodecahedron is longer than every
      // perpendicular width.
      const edges = [0, 1, 2].map((axis) => {
        const v = box[axis] as Vec3;
        return Math.hypot(v[0] as number, v[1] as number, v[2] as number);
      });
      const widths = perpendicularWidths(box) as unknown as number[];
      expect(Math.min(...widths)).toBeLessThan(Math.min(...edges));
      expect(0.5 * Math.min(...edges)).toBeGreaterThan(limit);
    },
  );

  it.skipIf(!present)("is still refused by the engine, and says so", async () => {
    const state = parseState(readFileSync(`${dodecahedron}/state.xml`, "utf8"));
    const system = parseSystem(readFileSync(`${dodecahedron}/system.xml`, "utf8"));
    const { evaluatePotential } = await import("../src/reference/potential.js");
    // The parsers take a triclinic box; the force path is what does not. Until
    // the rest of the chain exists, the refusal is the correct behaviour and
    // this test is what will fail when it is time to delete it.
    expect(() => evaluatePotential(system, state.positions, state.boxVectors)).toThrow(
      /triclinic/i,
    );
  });
});
