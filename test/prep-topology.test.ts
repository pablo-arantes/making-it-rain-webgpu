/**
 * Reading a PDB and reconstructing its bonds, against OpenMM's own topology.
 *
 * This is the first step of the preparation path and everything after it
 * depends on being right: template matching is graph matching, so a missing
 * peptide bond makes a chain look like twenty free amino acids and every
 * residue match the wrong template.
 *
 * Compared against `tools/dump_parameters.py`, which records the topology
 * OpenMM built from the same file -- atom by atom and bond by bond, not by
 * count. The same number of bonds can still be the wrong bonds.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";

interface Dump {
  readonly atoms: number;
  readonly topology: {
    readonly atoms: readonly [string, string, string, string, string | null][];
    readonly bonds: readonly (readonly [number, number])[];
  };
}

const pdbPath = "fixtures/prep/1L2Y.pdb";
const dumpPath = "fixtures/prep/1L2Y-params.json";
const present = existsSync(pdbPath) && existsSync(dumpPath);

describe.skipIf(!present)("the preparation path's topology", () => {
  const pdb = present ? readPdb(readFileSync(pdbPath, "utf8")) : undefined;
  const dump = present
    ? (JSON.parse(readFileSync(dumpPath, "utf8")) as Dump)
    : undefined;
  const table = parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8"));

  it("reads the same atoms, in the same order", () => {
    expect(pdb!.atoms).toHaveLength(dump!.atoms);
    const ours = pdb!.atoms.map((a) => [
      a.name,
      a.residueName,
      a.residueId,
      a.chainId,
      a.element,
    ]);
    const theirs = dump!.topology.atoms.map((a) => [a[0], a[1], a[2], a[3], a[4]]);
    // Element by element, so a difference names the atom.
    let differing = -1;
    for (let i = 0; i < theirs.length; i++) {
      if (JSON.stringify(ours[i]) !== JSON.stringify(theirs[i])) {
        differing = i;
        break;
      }
    }
    expect(
      differing,
      differing < 0
        ? ""
        : `atom ${differing}: ours ${JSON.stringify(ours[differing])}, OpenMM ${JSON.stringify(theirs[differing])}`,
    ).toBe(-1);
  });

  it("groups them into the same residues and chains", () => {
    const residues = new Set(dump!.topology.atoms.map((a) => `${a[3]}|${a[2]}|${a[1]}`));
    expect(pdb!.residues).toHaveLength(residues.size);
    expect(pdb!.chains).toHaveLength(new Set(dump!.topology.atoms.map((a) => a[3])).size);
  });

  it("reconstructs exactly OpenMM's bonds", () => {
    const ours = inferBonds(pdb!, table);
    const theirs = dump!.topology.bonds;
    const key = (pair: readonly [number, number]): string => `${pair[0]}-${pair[1]}`;
    const oursSet = new Set(ours.map(key));
    const theirsSet = new Set(theirs.map((p) => key(p)));
    const extra = [...oursSet].filter((k) => !theirsSet.has(k));
    const missing = [...theirsSet].filter((k) => !oursSet.has(k));
    const describe_ = (k: string): string => {
      const [a, b] = k.split("-").map(Number) as [number, number];
      const atom = (i: number): string => {
        const entry = dump!.topology.atoms[i] as [string, string, string, string, string | null];
        return `${entry[1]}${entry[2]}:${entry[0]}`;
      };
      return `${atom(a)}–${atom(b)}`;
    };
    expect(extra.map(describe_), "bonds we invented").toEqual([]);
    expect(missing.map(describe_), "bonds we missed").toEqual([]);
    expect(ours).toHaveLength(theirs.length);
  });

  it("makes the peptide bonds that join the chain", () => {
    // The `-`/`+` prefixes in residues.xml. Without them every residue is its
    // own molecule, which is the failure that makes template matching pick
    // free amino acids over chain residues.
    const ours = inferBonds(pdb!, table);
    const residueOf = new Map<number, number>();
    for (const [index, residue] of pdb!.residues.entries()) {
      for (const atom of residue.atoms) residueOf.set(atom, index);
    }
    const links = ours.filter(
      ([a, b]) => residueOf.get(a) !== residueOf.get(b),
    );
    // Twenty residues in one chain: nineteen peptide bonds and no more.
    expect(links).toHaveLength(pdb!.residues.length - 1);
    for (const [a, b] of links) {
      const names = [pdb!.atoms[a]!.name, pdb!.atoms[b]!.name].sort();
      expect(names).toEqual(["C", "N"]);
    }
  });

  it("reads the box when there is one, and nothing when there is not", () => {
    // This fixture is unsolvated, so it has no CRYST1 to read.
    expect(pdb!.box).toBeNull();
    const solvated = readPdb(readFileSync("fixtures/1L2Y/system.pdb", "utf8"));
    expect(solvated.box![0]).toBeCloseTo(3.0806, 3);
  });

  it("keeps a two-letter element distinct from a backbone atom", () => {
    // Fixed-width, and the field position carries meaning: ` CA ` is an alpha
    // carbon, `CA  ` is calcium. Splitting on whitespace loses that.
    const made = [
      "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C  ",
      "HETATM    2 CA    CA A   2       4.000   5.000   6.000  1.00  0.00          CA  ",
    ].join("\n");
    const read = readPdb(made);
    expect(read.atoms[0]!.element).toBe("C");
    expect(read.atoms[0]!.rawName).toBe(" CA ");
    expect(read.atoms[1]!.element).toBe("CA");
    expect(read.atoms[1]!.hetatm).toBe(true);
  });
});
