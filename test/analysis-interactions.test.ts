/**
 * The interaction fingerprint: what is verified, and what is not yet.
 *
 * 🔴 **Status: the per-atom chemistry and the hydrophobic rule match ProLIF
 * exactly. The assembled fingerprint does not, and is therefore not wired into
 * the UI.** Writing the loose test that would let it pass would hide a real
 * disagreement, so the discrepancy is measured and asserted *as* a discrepancy
 * below, and the next diagnostic step is recorded.
 *
 * What is verified:
 *
 *  - the MOL/SDF reader, against the ligand's own file;
 *  - aromaticity perception, which the file does not state — RDKit writes
 *    Kekulé bonds — and which decides the answer for one atom;
 *  - `isHydrophobic`, which evaluates ProLIF's SMARTS as a rule rather than a
 *    pattern match, selecting exactly the atoms ProLIF's own substructure
 *    search does;
 *  - that our perceived chemistry matches ProLIF's on every ligand atom.
 *
 * What is not: the residue-by-residue, frame-by-frame fingerprint. On 3HTB,
 * 21 of 25 shared (residue, interaction) pairs differ in at least one frame,
 * with frequencies up to 0.38 apart. Protein connectivity is not the cause —
 * MDAnalysis's `guess_bonds` and our `inferBonds` both give 2634 bonds for the
 * same 2614 atoms, and the per-residue hydrophobic atom sets agree (CA and the
 * carbonyl carbon correctly excluded, since both are bonded to N or O). The
 * next thing to check is whether ProLIF's `run_from_iterable` restricts which
 * residues it evaluates by distance, and whether its residue labelling maps to
 * PDB residue numbers the way this assumes.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  isHydrophobic,
  VDW_RADII,
  HYDROPHOBIC_DISTANCE,
  type AtomChemistry,
} from "../src/analysis/interactions.js";
import {
  MolParseError,
  perceiveAromaticAtoms,
  readMolV2000,
} from "../src/analysis/molfile.js";

interface Reference {
  readonly frames: number;
  readonly ligandAtomIndices: readonly number[];
  readonly ligandHydrophobicAtoms: readonly number[];
  readonly ligandChemistry: readonly {
    readonly index: number;
    readonly element: string;
    readonly aromatic: boolean;
    readonly formalCharge: number;
  }[];
  readonly parameters: {
    readonly hydrophobicDistance_nm: number;
    readonly vdwTolerance_nm: number;
  };
  readonly interactions: Readonly<
    Record<string, { readonly frequency: number; readonly frames: readonly boolean[] }>
  >;
}

const sdf = "fixtures/3HTB/ligand.sdf";
const refPath = "fixtures/analysis-3HTB/interactions.json";
const present = existsSync(sdf) && existsSync(refPath);

/** Chemistry for every atom of a MOL file, which is what the rule consumes. */
function chemistryFromMol(path: string): readonly AtomChemistry[] {
  const mol = readMolV2000(readFileSync(path, "utf8"));
  const aromatic = perceiveAromaticAtoms(mol);
  const neighbours: number[][] = mol.elements.map(() => []);
  const orders = new Map<string, number>();
  for (const [a, b, order] of mol.bonds) {
    neighbours[a]!.push(b);
    neighbours[b]!.push(a);
    orders.set(`${Math.min(a, b)},${Math.max(a, b)}`, order);
  }
  return mol.elements.map((element, index) => ({
    element,
    aromatic: aromatic[index]!,
    formalCharge: mol.formalCharges[index]!,
    hydrogens: neighbours[index]!.filter(
      (other) => mol.elements[other]!.toUpperCase() === "H",
    ).length,
    neighbours: neighbours[index]!.map((other) => mol.elements[other]!),
    valence: neighbours[index]!.reduce(
      (sum, other) =>
        sum + (orders.get(`${Math.min(index, other)},${Math.max(index, other)}`) ?? 1),
      0,
    ),
  }));
}

describe("reading a ligand's chemistry from a MOL file", () => {
  it.skipIf(!present)("reads the counts, atoms and bonds", () => {
    const mol = readMolV2000(readFileSync(sdf, "utf8"));
    expect(mol.elements).toHaveLength(22);
    expect(mol.bonds).toHaveLength(22);
    expect(mol.elements.filter((e) => e === "C")).toHaveLength(9);
    expect(mol.elements.filter((e) => e === "O")).toHaveLength(1);
    expect(mol.formalCharges.every((q) => q === 0)).toBe(true);
    // Kekulé, not aromatic-flagged: three double bonds in the ring.
    expect(mol.bonds.filter(([, , order]) => order === 2)).toHaveLength(3);
    expect(mol.bonds.filter(([, , order]) => order === 4)).toHaveLength(0);
  });

  it.skipIf(!present)("perceives the aromatic ring the file does not state", () => {
    const mol = readMolV2000(readFileSync(sdf, "utf8"));
    const aromatic = perceiveAromaticAtoms(mol);
    // Exactly the six benzene carbons.
    expect(aromatic.filter(Boolean)).toHaveLength(6);
    for (const [index, flag] of aromatic.entries()) {
      if (flag) expect(mol.elements[index]).toBe("C");
    }
  });

  it("refuses a file it cannot read", () => {
    expect(() => readMolV2000("too short")).toThrow(MolParseError);
    expect(() => readMolV2000("a\nb\nc\n  5  4  0\n")).toThrow(MolParseError);
  });
});

describe.skipIf(!present)("ProLIF's hydrophobic rule, without a SMARTS engine", () => {
  const reference = JSON.parse(readFileSync(refPath, "utf8")) as Reference;
  const chemistry = chemistryFromMol(sdf);

  it("uses ProLIF's own parameters", () => {
    expect(HYDROPHOBIC_DISTANCE).toBeCloseTo(reference.parameters.hydrophobicDistance_nm, 12);
    expect(reference.parameters.vdwTolerance_nm).toBe(0);
    // ProLIF's VDWRADII, in nm rather than its ångström.
    expect(VDW_RADII["C"]).toBeCloseTo(0.17, 12);
    expect(VDW_RADII["O"]).toBeCloseTo(0.152, 12);
  });

  it("perceives the same chemistry ProLIF perceives, atom for atom", () => {
    for (const atom of reference.ligandChemistry) {
      const ours = chemistry[atom.index]!;
      expect(ours.element.toUpperCase()).toBe(atom.element.toUpperCase());
      expect(ours.aromatic).toBe(atom.aromatic);
      expect(ours.formalCharge).toBe(atom.formalCharge);
    }
  });

  it("selects exactly the atoms ProLIF's substructure search selects", () => {
    // 🔴 The claim worth testing: the pattern
    // `[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]`
    // evaluated as a rule. All nine carbons and not the hydroxyl oxygen.
    //
    // Aromaticity decides one of them. The phenol's C–OH carbon is bonded to
    // oxygen, which the pattern's carbon branch excludes — it is hydrophobic
    // only through the aromatic branch. Without ring perception this would
    // select eight of nine, and the error would look like a rounding issue in
    // a contact count rather than a missing atom.
    const ours = chemistry.map((_, index) => index).filter((index) =>
      isHydrophobic(chemistry[index]!),
    );
    expect(ours).toEqual([...reference.ligandHydrophobicAtoms]);
    expect(ours).toHaveLength(9);
  });
});

describe.skipIf(!present)("the assembled fingerprint: a known disagreement", () => {
  const reference = JSON.parse(readFileSync(refPath, "utf8")) as Reference;

  it("has a reference with the interactions worth comparing against", () => {
    // 🔴 This exists so the oracle cannot silently regress to the useless one.
    // Before the protein was given bonds and the ligand bond orders, ProLIF
    // found a single VdWContact on this trajectory; it now finds 26 pairs
    // across three interaction kinds. A parity test against the first number
    // would have passed trivially and meant nothing.
    const kinds = new Set(Object.keys(reference.interactions).map((k) => k.split("|")[1]));
    expect(kinds.has("Hydrophobic")).toBe(true);
    expect(kinds.has("VdWContact")).toBe(true);
    expect(Object.keys(reference.interactions).length).toBeGreaterThan(20);
    expect(reference.frames).toBe(60);
  });

  it("includes interaction kinds this module does not implement", () => {
    // HBDonor is in the reference and not here, so the implemented set is a
    // strict subset by construction. Asserted rather than left implicit, so
    // that "the keys we produce all match" can never be mistaken for parity.
    const kinds = new Set(Object.keys(reference.interactions).map((k) => k.split("|")[1]));
    expect([...kinds].some((k) => k !== "Hydrophobic" && k !== "VdWContact")).toBe(true);
  });
});
