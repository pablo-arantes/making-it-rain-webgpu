/**
 * Template matching, and the per-atom parameters it unlocks.
 *
 * Compared against `tools/dump_parameters.py`'s record of what OpenMM assigned
 * to the same file: 304 masses and 304 (charge, sigma, epsilon) triples. That
 * comparison is stronger than it looks. Those four numbers together pin the
 * atom *type* for all but a handful of atoms, so matching the dump atom by atom
 * means the graph matcher chose the same template and the same correspondence
 * within it -- which is the thing that cannot be checked directly, because
 * OpenMM does not report which template it used.
 *
 * Charges are the sharpest part. ff14SB gives one type different charges in
 * different residues, so a charge that agrees for all 304 atoms means the
 * terminal variants were picked correctly; matching `NASN` as plain `ASN`
 * changes nine charges and nothing else.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { assignAtoms, totalCharge } from "../src/prep/assign.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";
import { parseForceField } from "../src/prep/ffxml.js";
import {
  matchResidues,
  prepareTemplates,
  residueSignature,
} from "../src/prep/template.js";

interface Dump {
  readonly atoms: number;
  readonly masses: readonly number[];
  readonly nonbonded: {
    /** charge (e), sigma (nm), epsilon (kJ/mol), per atom. */
    readonly particles: readonly (readonly [number, number, number])[];
  };
}

const pdbPath = "fixtures/prep/1L2Y.pdb";
const dumpPath = "fixtures/prep/1L2Y-params.json";
const fieldPath = "vendor/ffxml/protein.ff14SB.xml";
const present = existsSync(pdbPath) && existsSync(dumpPath) && existsSync(fieldPath);

describe.skipIf(!present)("assigning parameters to 1L2Y", () => {
  const topology = present ? readPdb(readFileSync(pdbPath, "utf8")) : undefined;
  const dump = present
    ? (JSON.parse(readFileSync(dumpPath, "utf8")) as Dump)
    : undefined;
  const field = present ? parseForceField(readFileSync(fieldPath, "utf8")) : undefined;
  const bonds = present
    ? inferBonds(
        topology!,
        parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
      )
    : undefined;

  it("matches every residue to exactly one template", () => {
    const matches = matchResidues(topology!, bonds!, field!);
    expect(matches).toHaveLength(topology!.residues.length);
    for (const [index, match] of matches.entries()) {
      expect(match.atomMatches).toHaveLength(
        topology!.residues[index]!.atoms.length,
      );
      // A permutation: every template atom used exactly once.
      expect(new Set(match.atomMatches).size).toBe(match.atomMatches.length);
    }
  });

  it("picks the terminal variants rather than the names the file gives", () => {
    const matches = matchResidues(topology!, bonds!, field!);
    const names = matches.map((m) => m.template);
    const residues = topology!.residues.map((r) => r.name);
    // 1L2Y is NLYIQWLKDGGPSSGRPPPS: asparagine first, serine last.
    expect(residues[0]).toBe("ASN");
    expect(residues[residues.length - 1]).toBe("SER");
    expect(names[0]).toBe("NASN");
    expect(names[names.length - 1]).toBe("CSER");
    // And nothing in between took a terminal template.
    for (const name of names.slice(1, -1)) {
      expect(name.startsWith("N")).toBe(name === "NLE" || name === "NME");
      expect(name.startsWith("C")).toBe(name === "CYS" || name === "CYX");
    }
  });

  it("assigns the masses OpenMM assigned", () => {
    const assignment = assignAtoms(topology!, bonds!, field!);
    expect(assignment.atoms).toHaveLength(dump!.atoms);
    const ours = assignment.atoms.map((a) => a.mass);
    // The ffxml writes masses to two decimals and OpenMM reads them straight
    // through, so this is an exact comparison, not a tolerance.
    expect(ours).toEqual([...dump!.masses]);
  });

  it("assigns the charges OpenMM assigned, atom by atom", () => {
    const assignment = assignAtoms(topology!, bonds!, field!);
    const expected = dump!.nonbonded.particles.map((p) => p[0]);
    const ours = assignment.atoms.map((a) => a.charge);
    for (const [index, charge] of ours.entries()) {
      // 1e-12 rather than exact: the dump round-trips through JSON, and the
      // template attribute is a decimal string either way.
      expect(charge).toBeCloseTo(expected[index] as number, 12);
    }
  });

  it("assigns the Lennard-Jones parameters OpenMM assigned", () => {
    const assignment = assignAtoms(topology!, bonds!, field!);
    for (const [index, atom] of assignment.atoms.entries()) {
      const [, sigma, epsilon] = dump!.nonbonded.particles[index] as readonly [
        number,
        number,
        number,
      ];
      // 🔴 Not exact, and this is the reason: the ffxml gives sigma directly,
      // but OpenMM's own amber14 set is loaded through the same path, so a
      // disagreement here would be a unit or a radius/diameter confusion
      // rather than round-off. 1e-10 catches both while tolerating JSON.
      expect(atom.sigma).toBeCloseTo(sigma, 10);
      expect(atom.epsilon).toBeCloseTo(epsilon, 10);
    }
  });

  /**
   * 🔴 This test asserts what the *fixture exercises*, not what the code
   * returns, and it exists because the first version of this file passed six
   * for six and proved less than it looked.
   *
   * Both of 1L2Y's terminal residues differ from their mid-chain templates in
   * atom count, so the element signature alone separates them -- the
   * backtracking search and the external-bond test could both have been broken
   * with every other test still green. They are not, as it turns out: ff14SB
   * has twelve signature groups holding two templates each, and this peptide
   * hits three of them. Measured:
   *
   *   LEU/ILE/LEU (residues 1, 3, 6)   2 candidates, identical external bonds
   *                                    -> only the bond graph can decide
   *   TYR vs CPHE (residue 2)          settled by the external-bond count
   *   SER vs CALA (residues 12, 13)    settled by the external-bond count
   *
   * ILE and LEU are the valuable case: same formula, same two external bonds,
   * different branching. Getting those right is the backtracking search
   * working, and the charge comparison above is what confirms it.
   *
   * If a future change swaps this fixture for one without an ILE/LEU pair, this
   * fails and says what was lost, rather than quietly reducing the matcher to
   * an atom counter.
   */
  it("exercises both the bond graph and the external-bond test", () => {
    const prepared = prepareTemplates(field!);
    const neighbours: number[][] = topology!.atoms.map(() => []);
    for (const [a, b] of bonds!) {
      neighbours[a]!.push(b);
      neighbours[b]!.push(a);
    }

    let graphOnly = 0;
    let byExternal = 0;
    for (const residue of topology!.residues) {
      const inside = new Set(residue.atoms);
      const elements = residue.atoms.map((a) => topology!.atoms[a]!.element);
      const candidates = prepared.get(residueSignature(elements)) ?? [];
      if (candidates.length < 2) continue;
      const external = residue.atoms.map(
        (atom) => neighbours[atom]!.filter((other) => !inside.has(other)).length,
      );
      const key = [...external].sort().join(",");
      const survivors = candidates.filter(
        (c) => [...c.externalBonds].sort().join(",") === key,
      );
      if (survivors.length === 1) byExternal++;
      else graphOnly++;
    }

    // Three ILE/LEU pairs the graph alone decides, three the external-bond
    // count decides. Exact, because a change in either number means the
    // fixture no longer covers what the comment says it covers.
    expect(graphOnly).toBe(3);
    expect(byExternal).toBe(3);

    const matches = matchResidues(topology!, bonds!, field!);
    const names = matches.map((m) => m.template);
    expect([names[1], names[3], names[6]]).toEqual(["LEU", "ILE", "LEU"]);
  });

  it("gives the peptide an integer total charge", () => {
    const assignment = assignAtoms(topology!, bonds!, field!);
    const total = totalCharge(assignment);
    // 1L2Y: +1 at the N terminus, -1 at the C terminus, LYS +1, ARG +1,
    // ASP -1 -- net +1. A non-integer here means a template was matched with
    // the wrong charge set, which is exactly what a name-based matcher does.
    expect(total).toBeCloseTo(1, 6);
    const fromDump = dump!.nonbonded.particles.reduce((sum, p) => sum + p[0], 0);
    expect(total).toBeCloseTo(fromDump, 10);
  });
});
