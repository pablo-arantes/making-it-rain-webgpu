/**
 * Position restraints against OpenMM's own `CustomExternalForce`.
 *
 * Three configurations, and the third is the one that matters. At the reference
 * positions the energy is exactly zero and every force vanishes, so that case
 * alone would pass an implementation that computed nothing. A small
 * displacement then exercises the arithmetic. A displacement larger than half
 * the box exercises the **minimum image**, which is the one part of this force
 * that can be wrong invisibly: without the wrap, every atom that has not
 * crossed a boundary gives an identical energy, so a freshly minimised
 * structure looks perfect right up until something drifts across and gets
 * pulled by `k` times a box length.
 *
 * The expected forces come from the restraint's **own force group**, so they are
 * the restraint alone rather than a difference of two large nearly-cancelling
 * totals.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseSystem, type PositionRestraintForce } from "../src/engine/system.js";
import { positionRestraint } from "../src/reference/forces/restraint.js";
import { rectangularBox } from "../src/reference/forces/nonbonded.js";
import { readNpy } from "./npy.js";
import { FIXTURE_NAMES } from "./fixtures.js";

interface RestraintFacts {
  readonly restrained_atoms: number;
  readonly k_kJ_mol_nm2: number;
  readonly cases: Record<
    string,
    {
      readonly displacement_nm: number;
      readonly energy_kJ_mol: number;
      readonly force_rms: number;
      readonly max_force_component: number;
    }
  >;
}

for (const name of FIXTURE_NAMES) {
  const dir = `fixtures/${name}`;
  const present = existsSync(`${dir}/restraint-facts.json`);

  describe.skipIf(!present)(`position restraints for ${name}`, () => {
    const facts = present
      ? (JSON.parse(readFileSync(`${dir}/restraint-facts.json`, "utf8")) as RestraintFacts)
      : undefined;
    const system = present
      ? parseSystem(readFileSync(`${dir}/system-restrained.xml`, "utf8"))
      : undefined;
    const restraint = system?.forces.find(
      (f): f is PositionRestraintForce => f.type === "PositionRestraint",
    );

    it("parses the one expression the engine implements", () => {
      expect(restraint).toBeDefined();
      expect(restraint!.count).toBe(facts!.restrained_atoms);
      // Whitespace is removed when parsing, because OpenMM writes the
      // expression with spaces and the engine compares it literally.
      expect(restraint!.energyExpression.replace(/\s+/g, "")).toBe(
        "k*periodicdistance(x,y,z,x0,y0,z0)^2",
      );
      for (let i = 0; i < restraint!.count; i++) {
        expect(restraint!.k[i]).toBe(facts!.k_kJ_mol_nm2);
      }
    });

    it.each(["reference", "displaced", "wrapped"])(
      "matches OpenMM's energy and forces, %s",
      (label) => {
        const expected = facts!.cases[label]!;
        const positions = readNpy(`${dir}/restraint-${label}-positions.npy`).data;
        const openmmForces = readNpy(`${dir}/restraint-${label}-forces.npy`).data;
        const forces = new Float64Array(system!.particleCount * 3);
        const energy = positionRestraint(
          restraint!,
          positions,
          system!.boxVectors,
          forces,
        );

        let worst = 0;
        for (let i = 0; i < forces.length; i++) {
          worst = Math.max(
            worst,
            Math.abs((forces[i] as number) - (openmmForces[i] as number)),
          );
        }
        // f64 against f64 through the same algebra, so this is round-off and
        // nothing else: the tolerance is absolute because at the reference
        // positions the expected values are exactly zero and a relative one
        // would be undefined.
        expect(Math.abs(energy - expected.energy_kJ_mol)).toBeLessThan(1e-9);
        expect(worst).toBeLessThan(1e-9);
      },
    );

    it("applies the minimum image rather than tethering to one copy", () => {
      // The decisive check, stated as a bound rather than a comparison: the
      // restraint force cannot exceed `2 k` times half a box diagonal, because
      // no atom is ever further than that from the nearest image of its
      // reference point. An implementation without the wrap breaks this as soon
      // as anything moves a box length.
      const positions = readNpy(`${dir}/restraint-wrapped-positions.npy`).data;
      const forces = new Float64Array(system!.particleCount * 3);
      positionRestraint(restraint!, positions, system!.boxVectors, forces);
      const box = rectangularBox(system!.boxVectors);
      const halfDiagonal =
        0.5 *
        Math.hypot(box[0] as number, box[1] as number, box[2] as number);
      const bound = 2 * facts!.k_kJ_mol_nm2 * halfDiagonal;
      let worst = 0;
      for (const value of forces) worst = Math.max(worst, Math.abs(value));
      expect(worst).toBeLessThanOrEqual(bound);
      // And it has to be doing *something* there, or the bound is vacuous.
      expect(worst).toBeGreaterThan(0.1 * bound);
    });
  });
}
