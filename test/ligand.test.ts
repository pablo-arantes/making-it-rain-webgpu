/**
 * Finding a ligand and giving it its chemistry, against RDKit.
 *
 * A PDB does not say what a ligand *is*: `HETATM` records carry an element, a
 * name and coordinates, and no bonds at all. Everything Phase 4 does afterwards
 * — typing, protonating, charging — needs the chemistry, and the only authority
 * for a three-letter code is the RCSB Chemical Component Dictionary.
 *
 * 🔴 The claim under test is that **transferring bond orders by atom name gives
 * the same answer as RDKit's substructure match**. The Python route uses
 * `AssignBondOrdersFromTemplate`, which the browser cannot run; the names in a
 * deposited `HETATM` record are the CCD's own `atom_id` values, so the
 * correspondence is already in the file. If that holds, the browser needs no
 * graph matcher for this step — and it is also *safer*, because a substructure
 * match has to choose among symmetry-equivalent mappings and for a symmetric
 * ligand that choice is arbitrary.
 *
 * Both ligands the plan's gate names: JZ4 from 3HTB (propylphenol — a
 * substituted aromatic with a hydroxyl, so Kekulé bonds, an aromatic flag and
 * a heteroatom) and benzene from 181L (six-fold symmetric, which is the case
 * where a substructure match has six equally good answers).
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { CcdParseError, parseChemicalComponent } from "../src/prep/ccd.js";
import {
  findLigandCandidates,
  implicitHydrogens,
  LigandMatchFailed,
  matchLigand,
} from "../src/prep/ligand.js";

interface Reference {
  readonly smiles: string;
  readonly formalCharge: number;
  readonly heavyAtoms: number;
  readonly atomsWithHydrogens: number;
  readonly bonds: readonly {
    readonly atoms: readonly [string, string];
    readonly order: string;
    readonly aromatic: boolean;
  }[];
  readonly aromaticAtoms: readonly string[];
  readonly hydrogenCounts: Readonly<Record<string, number>>;
}

const CASES = [
  {
    code: "JZ4",
    structure: "fixtures/3HTB/3HTB.pdb",
    cif: "fixtures/ligand/JZ4.cif",
    reference: "fixtures/ligand/JZ4.json",
    heavy: 10,
    note: "propylphenol: Kekulé aromatic ring plus a hydroxyl",
  },
  {
    code: "BNZ",
    structure: "fixtures/ligand/181L.pdb",
    cif: "fixtures/ligand/BNZ.cif",
    reference: "fixtures/ligand/BNZ.json",
    heavy: 6,
    note: "benzene: six-fold symmetric, where a substructure match is ambiguous",
  },
] as const;

for (const entry of CASES) {
  const present =
    existsSync(entry.structure) && existsSync(entry.cif) && existsSync(entry.reference);

  describe.skipIf(!present)(`${entry.code} (${entry.note})`, () => {
    const topology = present ? readPdb(readFileSync(entry.structure, "utf8")) : undefined;
    const component = present
      ? parseChemicalComponent(readFileSync(entry.cif, "utf8"))
      : undefined;
    const reference = present
      ? (JSON.parse(readFileSync(entry.reference, "utf8")) as Reference)
      : undefined;
    const candidate = present
      ? findLigandCandidates(topology!).find((c) => c.code === entry.code)
      : undefined;
    const matched =
      present && candidate !== undefined
        ? matchLigand(topology!, candidate.residue, component!)
        : undefined;

    it("finds the ligand among the water, ions and additives", () => {
      const candidates = findLigandCandidates(topology!);
      expect(candidate, `${entry.code} was not found at all`).toBeDefined();
      expect(candidate!.heavyAtoms).toBe(entry.heavy);
      expect(candidate!.likelyAdditive).toBe(false);
      // Ranked ahead of everything flagged as an additive. 3HTB has PO4 and
      // BME; 181L has chloride and HED.
      const ranked = candidates.findIndex((c) => c.code === entry.code);
      const firstAdditive = candidates.findIndex((c) => c.likelyAdditive);
      expect(ranked).toBeLessThan(firstAdditive);
      // Nothing is hidden, though: the additives are still listed, because a
      // structure can have glycerol as its real ligand.
      expect(candidates.some((c) => c.code === "HOH")).toBe(true);
    });

    it("transfers the bond orders RDKit's substructure match produces", () => {
      const name = (atom: number): string =>
        (topology!.atoms[atom] as { name: string }).name;
      const isHydrogen = (atom: number): boolean =>
        (topology!.atoms[atom] as { element: string }).element.toUpperCase() === "H";
      const ours = matched!.bonds
        .filter((bond) => !isHydrogen(bond.atoms[0]) && !isHydrogen(bond.atoms[1]))
        .map((bond) => {
          const [a, b] = [name(bond.atoms[0]), name(bond.atoms[1])].sort();
          return `${a}-${b}:${bond.aromatic ? "AROM" : bond.order}`;
        })
        .sort();
      const theirs = reference!.bonds
        .map((bond) => `${bond.atoms[0]}-${bond.atoms[1]}:${bond.aromatic ? "AROM" : bond.order}`)
        .sort();
      expect(ours).toEqual(theirs);
    });

    it("agrees on which atoms are aromatic and on the formal charge", () => {
      expect(matched!.aromatic.filter(Boolean)).toHaveLength(
        reference!.aromaticAtoms.length,
      );
      expect(matched!.totalCharge).toBe(reference!.formalCharge);
    });

    it("derives the hydrogen count RDKit derives, from valence alone", () => {
      // The step that makes the molecule typable: a deposited ligand has no
      // hydrogens, and how many each heavy atom needs follows from its bond
      // orders and formal charge.
      for (const [slot, atom] of matched!.atoms.entries()) {
        const element = (topology!.atoms[atom] as { element: string }).element;
        if (element.toUpperCase() === "H") continue;
        const orders = matched!.bonds
          .filter((bond) => bond.atoms[0] === atom || bond.atoms[1] === atom)
          .map((bond) => (bond.aromatic ? ("AROM" as const) : bond.order));
        const got = implicitHydrogens(element, orders, matched!.formalCharges[slot]!);
        const want = reference!.hydrogenCounts[
          (topology!.atoms[atom] as { name: string }).name
        ];
        if (want === undefined) continue;
        expect(got, `hydrogen count for ${(topology!.atoms[atom] as { name: string }).name}`).toBe(want);
      }
      // And they add up to RDKit's total.
      const total = matched!.atoms.reduce((sum, atom, slot) => {
        const element = (topology!.atoms[atom] as { element: string }).element;
        if (element.toUpperCase() === "H") return sum;
        const orders = matched!.bonds
          .filter((bond) => bond.atoms[0] === atom || bond.atoms[1] === atom)
          .map((bond) => (bond.aromatic ? ("AROM" as const) : bond.order));
        return sum + implicitHydrogens(element, orders, matched!.formalCharges[slot]!);
      }, 0);
      expect(total + entry.heavy).toBe(reference!.atomsWithHydrogens);
    });

    it("reports missing hydrogens as routine and missing heavy atoms as not", () => {
      // 🔴 An undifferentiated "12 atoms missing" is useless: a deposited
      // ligand almost never has hydrogens. Only a missing *heavy* atom means
      // the model is not the component it claims to be.
      expect(matched!.missingHeavyAtoms).toBe(0);
      expect(matched!.missing.length).toBeGreaterThan(0);
      for (const atom of matched!.missing) {
        expect(atom.element.toUpperCase()).toBe("H");
      }
      expect(matched!.unexpected).toEqual([]);
    });
  });
}

describe("the CCD parser", () => {
  it.skipIf(!existsSync("fixtures/ligand/JZ4.cif"))("reads both loops", () => {
    const component = parseChemicalComponent(
      readFileSync("fixtures/ligand/JZ4.cif", "utf8"),
    );
    expect(component.id).toBe("JZ4");
    // 22 atoms including the twelve hydrogens the deposited fragment lacks.
    expect(component.atoms).toHaveLength(22);
    expect(component.bonds).toHaveLength(22);
    expect(component.atoms.filter((a) => a.aromatic)).toHaveLength(6);
    // The CCD stores an aromatic ring as Kekulé single and double bonds *plus*
    // an aromatic flag, so both have to be read -- the orders alone would make
    // the ring look aliphatic and the flag alone would lose the alternation.
    const ring = component.bonds.filter((b) => b.aromatic);
    expect(ring).toHaveLength(6);
    expect(ring.filter((b) => b.order === "DOUB")).toHaveLength(3);
    expect(ring.filter((b) => b.order === "SING")).toHaveLength(3);
  });

  it("refuses what it cannot read rather than guessing", () => {
    expect(() => parseChemicalComponent("nothing here")).toThrow(CcdParseError);
    expect(() =>
      parseChemicalComponent("data_XXX\nloop_\n_chem_comp_atom.atom_id\n"),
    ).toThrow(CcdParseError);
  });

  it.skipIf(!existsSync("fixtures/ligand/JZ4.cif"))(
    "refuses a residue that is not the component",
    () => {
      const component = parseChemicalComponent(
        readFileSync("fixtures/ligand/JZ4.cif", "utf8"),
      );
      const topology = readPdb(readFileSync("fixtures/3HTB/3HTB.pdb", "utf8"));
      const water = findLigandCandidates(topology).find((c) => c.code === "HOH")!;
      expect(() => matchLigand(topology, water.residue, component)).toThrow(
        LigandMatchFailed,
      );
    },
  );
});
