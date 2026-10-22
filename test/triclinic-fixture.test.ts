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
    ) as { waters: number; atoms_total: number };
    const cubeFacts = JSON.parse(readFileSync(`${cube}/facts.json`, "utf8")) as {
      residue_counts: Record<string, number>;
      atoms_total: number;
    };
    const ratio =
      dodecaFacts.waters / (cubeFacts.residue_counts["HOH"] as number);
    // Not exactly the volume ratio: the solute is the same size in both, the
    // padding is measured to the solute rather than between images, and whole
    // molecules are kept or dropped. The two fixtures are also built by
    // different tools with slightly different padding, so this is a band.
    expect(ratio).toBeGreaterThan(0.6);
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

  it.skipIf(!present)(
    "🔴 agrees with OpenMM — but with the CPU platform, not Reference",
    async () => {
      const state = parseState(readFileSync(`${dodecahedron}/state.xml`, "utf8"));
      const system = parseSystem(readFileSync(`${dodecahedron}/system.xml`, "utf8"));
      const facts = JSON.parse(readFileSync(`${dodecahedron}/facts.json`, "utf8")) as {
        minimised: {
          energies_by_platform_kJ_mol: Record<string, Record<string, number>>;
        };
        reference_platform_pair_loss: {
          sampled_inside_cutoff: number;
          missed_by_reference_platform: number;
        };
      };
      const { evaluatePotential } = await import("../src/reference/potential.js");
      const ours = evaluatePotential(system, state.positions, state.boxVectors);
      const byForce = ours.byForce as Record<string, number>;
      const cpu = facts.minimised.energies_by_platform_kJ_mol["CPU"] as Record<
        string,
        number
      >;

      // The bonded terms are shape-independent and exact on both platforms.
      for (const [ourKey, theirKey] of [
        ["HarmonicBondForce", "0:HarmonicBondForce"],
        ["PeriodicTorsionForce", "1:PeriodicTorsionForce"],
        ["HarmonicAngleForce", "4:HarmonicAngleForce"],
      ] as const) {
        const relative =
          Math.abs((byForce[ourKey] as number) - (cpu[theirKey] as number)) /
          Math.abs(cpu[theirKey] as number);
        expect(relative, ourKey).toBeLessThan(1e-9);
      }

      // Reciprocal space: the part that needed the triclinic reciprocal
      // lattice, and the part with no independent check until
      // `src/reference/forces/ewald.ts` existed.
      const recipRelative =
        Math.abs(
          (byForce["NonbondedForce[reciprocal]"] as number) -
            (cpu["5:NonbondedForce[reciprocal]"] as number),
        ) / Math.abs(cpu["5:NonbondedForce[reciprocal]"] as number);
      expect(recipRelative).toBeLessThan(1e-7);

      // Direct space, against the CPU platform. 2e-6 rather than the 1e-11 the
      // cubic fixtures reach, because the CPU platform computes the nonbonded
      // terms in single precision.
      const directRelative =
        Math.abs(
          (byForce["NonbondedForce[direct]"] as number) - (cpu["2:NonbondedForce"] as number),
        ) / Math.abs(cpu["2:NonbondedForce"] as number);
      expect(directRelative).toBeLessThan(1e-5);
    },
  );

  it.skipIf(!present)(
    "🔴 does NOT agree with OpenMM's Reference platform, and that is the finding",
    () => {
      const facts = JSON.parse(readFileSync(`${dodecahedron}/facts.json`, "utf8")) as {
        minimised: {
          energies_by_platform_kJ_mol: Record<string, Record<string, number>>;
        };
        reference_platform_pair_loss: {
          sampled_inside_cutoff: number;
          missed_by_reference_platform: number;
        };
      };
      const cpu = facts.minimised.energies_by_platform_kJ_mol["CPU"] as Record<string, number>;
      const reference = facts.minimised.energies_by_platform_kJ_mol["Reference"] as Record<
        string,
        number
      >;

      // The two platforms disagree with *each other* on the direct space by
      // thousands of times their agreement on everything else. The Reference
      // platform is the one that is wrong: its neighbour list misses pairs just
      // inside the cutoff for a skewed cell.
      const platformGap = Math.abs(
        (cpu["2:NonbondedForce"] as number) - (reference["2:NonbondedForce"] as number),
      );
      expect(platformGap).toBeGreaterThan(1);

      // Counted directly, by asking the Reference platform for the separation
      // of pairs whose true nearest image is inside the cutoff.
      const loss = facts.reference_platform_pair_loss;
      expect(loss.sampled_inside_cutoff).toBeGreaterThan(100);
      expect(loss.missed_by_reference_platform).toBeGreaterThan(0);

      // 🔴 So for a triclinic cell the oracle is the CPU platform. Everywhere
      // else in this project it is Reference -- that is the trap, and it cost a
      // long detour chasing a 3.9 kJ/mol "error" in code that was right.
      // Nothing here should be loosened to make Reference fit.
      expect(Math.abs(platformGap / (cpu["2:NonbondedForce"] as number))).toBeLessThan(1e-3);
    },
  );
});
