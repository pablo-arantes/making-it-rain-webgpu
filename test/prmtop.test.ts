/**
 * Amber `prmtop` and `inpcrd`, verified by energy rather than by parsing.
 *
 * 🔴 "Does it load" is not a test of this format. A prmtop with an exclusion
 * window off by one, a double-counted 1-4, or a Lennard-Jones type matrix
 * transposed parses perfectly and gives a plausible number. So the check is
 * `tools/check_prmtop.py`: OpenMM reads the file this project wrote, builds a
 * `System` from it, and the per-force-group energies are compared against the
 * `System` the file was written from.
 *
 * That comparison caught the one real bug here, and nothing else would have.
 * The bonded terms matched to 1e-10 while the nonbonded energy was out by 2791
 * kJ/mol on a 304-atom peptide — see `src/prep/prmtop.ts` on zero-constant
 * dihedrals.
 *
 * Two fixtures, because they exercise different halves: the unsolvated peptide
 * has no box and no water, and the solvated one has both plus ions.
 *
 * The residual ~1e-8 is the round trip through kcal/mol. A prmtop stores
 * kcal/mol in `%FORMAT(5E16.8)`, nine significant figures, and OpenMM converts
 * back to kJ/mol on the way in.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { parseSystem } from "../src/engine/system.js";
import {
  AMBER_CHARGE_SCALE,
  PrmtopUnsupported,
  writeInpcrd,
  writePrmtop,
} from "../src/prep/prmtop.js";

interface Energies {
  readonly particles: number;
  readonly periodic: boolean;
  readonly fromPrmtop: Readonly<Record<string, number>>;
  readonly fromSystem: Readonly<Record<string, number>>;
  readonly totalFromPrmtop: number;
  readonly totalFromSystem: number;
}

const CASES = [
  {
    name: "1L2Y",
    energies: "fixtures/amber/1L2Y-energies.json",
    note: "an unsolvated peptide: no box, no water",
  },
  {
    name: "solvated",
    energies: "fixtures/amber/solvated-energies.json",
    note: "a solvated box with ions",
  },
] as const;

for (const entry of CASES) {
  describe.skipIf(!existsSync(entry.energies))(
    `the prmtop for ${entry.name} (${entry.note})`,
    () => {
      const energies = JSON.parse(readFileSync(entry.energies, "utf8")) as Energies;

      it("gives OpenMM the same energy per force group as the System it came from", () => {
        for (const name of Object.keys(energies.fromSystem)) {
          const theirs = energies.fromSystem[name] as number;
          const ours = energies.fromPrmtop[name];
          expect(ours, `${name} missing from the prmtop's system`).toBeDefined();
          expect(
            Math.abs((ours as number) - theirs) / Math.max(1, Math.abs(theirs)),
          ).toBeLessThan(1e-6);
        }
      });

      it("gives the same total", () => {
        expect(
          Math.abs(energies.totalFromPrmtop - energies.totalFromSystem) /
            Math.abs(energies.totalFromSystem),
        ).toBeLessThan(1e-6);
      });

      it("has the nonbonded group right, which is where the 1-4s live", () => {
        // Called out separately because this is the term the zero-constant
        // dihedral bug moved, and the one a wrong exclusion list moves. On
        // 1L2Y it was out by 2791 kJ/mol against a -2431 kJ/mol true value --
        // more than 100% -- while every other group was exact.
        const theirs = energies.fromSystem["NonbondedForce"] as number;
        const ours = energies.fromPrmtop["NonbondedForce"] as number;
        expect(Math.abs(ours - theirs) / Math.abs(theirs)).toBeLessThan(1e-7);
      });
    },
  );
}

describe("writing a prmtop", () => {
  const present = existsSync("fixtures/prep/1L2Y-system.xml");

  it.skipIf(!present)("writes the sections Amber's readers require", () => {
    const system = parseSystem(readFileSync("fixtures/prep/1L2Y-system.xml", "utf8"));
    const topology = readPdb(readFileSync("fixtures/prep/1L2Y.pdb", "utf8"));
    const text = writePrmtop(system, topology);
    for (const flag of [
      "POINTERS", "ATOM_NAME", "CHARGE", "ATOMIC_NUMBER", "MASS",
      "ATOM_TYPE_INDEX", "NUMBER_EXCLUDED_ATOMS", "NONBONDED_PARM_INDEX",
      "RESIDUE_LABEL", "RESIDUE_POINTER", "BOND_FORCE_CONSTANT",
      "DIHEDRAL_PERIODICITY", "SCEE_SCALE_FACTOR", "SCNB_SCALE_FACTOR",
      "LENNARD_JONES_ACOEF", "EXCLUDED_ATOMS_LIST", "AMBER_ATOM_TYPE",
    ]) {
      expect(text, `missing %FLAG ${flag}`).toContain(`%FLAG ${flag}`);
    }
    // Every line of a prmtop is 80 columns or less, and the %FLAG/%FORMAT
    // lines are padded to exactly 80 -- some readers rely on it.
    for (const line of text.split("\n")) expect(line.length).toBeLessThanOrEqual(80);
  });

  it.skipIf(!present)("scales charge the way Amber stores it", () => {
    // 🔴 A prmtop holds charge in units of e * 18.2223, not in e. Getting this
    // wrong scales the whole electrostatic energy by 332, which is obvious --
    // unlike most ways of being wrong here.
    expect(AMBER_CHARGE_SCALE).toBeCloseTo(18.2223, 10);
    const system = parseSystem(readFileSync("fixtures/prep/1L2Y-system.xml", "utf8"));
    const topology = readPdb(readFileSync("fixtures/prep/1L2Y.pdb", "utf8"));
    const text = writePrmtop(system, topology);
    const chargeBlock = text.slice(text.indexOf("%FLAG CHARGE"));
    const firstValue = Number(
      chargeBlock.split("\n")[2]!.slice(0, 16).trim().replace("E", "e"),
    );
    const nonbonded = system.forces.find((f) => f.type === "NonbondedForce") as {
      charge: Float64Array;
    };
    expect(firstValue).toBeCloseTo((nonbonded.charge[0] as number) * AMBER_CHARGE_SCALE, 6);
  });

  it("refuses a system it cannot express", () => {
    // A force with no Amber equivalent would be dropped, and a file missing a
    // term reads as valid. Refused by name instead.
    const system = parseSystem(readFileSync("fixtures/prep/1L2Y-system.xml", "utf8"));
    const topology = readPdb(readFileSync("fixtures/prep/1L2Y.pdb", "utf8"));
    const withExtra = {
      ...system,
      forces: [
        ...system.forces,
        { type: "MonteCarloBarostat", forceGroup: 0, name: "b" } as never,
      ],
    };
    expect(() => writePrmtop(withExtra, topology)).toThrow(PrmtopUnsupported);
    // And a topology that is not the same system.
    expect(() =>
      writePrmtop(system, { ...topology, atoms: topology.atoms.slice(0, 10) }),
    ).toThrow(PrmtopUnsupported);
  });

  it.skipIf(!present)("writes an inpcrd in ångström, six values a line", () => {
    const topology = readPdb(readFileSync("fixtures/prep/1L2Y.pdb", "utf8"));
    const text = writeInpcrd(topology.positions, { box: [3, 3, 3] });
    const lines = text.trim().split("\n");
    expect(Number(lines[1])).toBe(topology.atoms.length);
    // Second line of coordinates: six fields of twelve characters.
    expect(lines[2]).toHaveLength(72);
    const first = Number(lines[2]!.slice(0, 12));
    expect(first).toBeCloseTo((topology.positions[0] as number) * 10, 5);
    // The box, as three lengths then three angles.
    const last = lines[lines.length - 1]!;
    expect(Number(last.slice(0, 12))).toBeCloseTo(30, 5);
    expect(Number(last.slice(36, 48))).toBeCloseTo(90, 5);
  });
});
