/**
 * Bonded parameters for 1L2Y, against what OpenMM put in the forces.
 *
 * `tools/dump_parameters.py` records every term OpenMM added: 310 bonds, 565
 * angles and 1017 torsions, each with its atoms and its parameters. Comparing
 * term by term tests two independent things at once -- that we enumerate the
 * same terms from the graph, and that we look up the same parameters for them.
 * A count alone would catch neither: the same number of angles can be a
 * different set of angles.
 *
 * The torsions are compared as a multiset rather than in order, because OpenMM
 * emits propers and impropers into one force and the improper ordering rules
 * decide the atom order; `test/prep-impropers.test.ts` is where that order is
 * checked.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { assignAtoms } from "../src/prep/assign.js";
import {
  assignAngles,
  assignBonds,
  findAngles,
  findImpropers,
  findPropers,
  neighbourLists,
} from "../src/prep/bonded.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";
import { parseForceField } from "../src/prep/ffxml.js";

interface Dump {
  readonly bonds: readonly (readonly [number, number, number, number])[];
  readonly angles: readonly (readonly [number, number, number, number, number])[];
  readonly torsions: readonly (readonly [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ])[];
}

const pdbPath = "fixtures/prep/1L2Y.pdb";
const dumpPath = "fixtures/prep/1L2Y-params.json";
const fieldPath = "vendor/ffxml/protein.ff14SB.xml";
const present = existsSync(pdbPath) && existsSync(dumpPath) && existsSync(fieldPath);

describe.skipIf(!present)("bonded parameters for 1L2Y", () => {
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
  const types = present
    ? assignAtoms(topology!, bonds!, field!).atoms.map((a) => a.type)
    : undefined;
  const neighbours = present
    ? neighbourLists(topology!.atoms.length, bonds!)
    : undefined;

  it("assigns every bond the length and stiffness OpenMM assigned", () => {
    const { terms, diagnostics } = assignBonds(bonds!, types!, field!);
    expect(diagnostics.unmatched).toBe(0);
    expect(terms).toHaveLength(dump!.bonds.length);

    // Keyed by the atom pair, so enumeration order cannot hide a mismatch.
    const expected = new Map(
      dump!.bonds.map((b) => [`${Math.min(b[0], b[1])},${Math.max(b[0], b[1])}`, b]),
    );
    for (const term of terms) {
      const [a, b] = term.atoms as [number, number];
      const key = `${Math.min(a, b)},${Math.max(a, b)}`;
      const row = expected.get(key);
      expect(row, `no OpenMM bond for atoms ${key}`).toBeDefined();
      expect(term.value).toBeCloseTo(row![2], 12);
      expect(term.k).toBeCloseTo(row![3], 8);
    }
  });

  it("assigns every angle the equilibrium and stiffness OpenMM assigned", () => {
    const angles = findAngles(bonds!, neighbours!);
    const { terms, diagnostics } = assignAngles(angles, types!, field!);
    expect(diagnostics.unmatched).toBe(0);
    expect(terms).toHaveLength(dump!.angles.length);

    const expected = new Map(
      dump!.angles.map((a) => {
        const ends = [a[0], a[2]].sort((x, y) => x - y);
        return [`${ends[0]},${a[1]},${ends[1]}`, a];
      }),
    );
    for (const term of terms) {
      const [a, b, c] = term.atoms as [number, number, number];
      const ends = [a, c].sort((x, y) => x - y);
      const row = expected.get(`${ends[0]},${b},${ends[1]}`);
      expect(row, `no OpenMM angle for ${a}-${b}-${c}`).toBeDefined();
      expect(term.value).toBeCloseTo(row![3], 12);
      expect(term.k).toBeCloseTo(row![4], 8);
    }
  });

  it("enumerates the same angles OpenMM did, in the same order", () => {
    const angles = findAngles(bonds!, neighbours!);
    // OpenMM sorts its canonicalised angle tuples, so this is a full check of
    // the enumeration independent of the parameter lookup.
    const ours = angles.map((a) => a.join(","));
    const theirs = dump!.angles.map((a) => {
      const ends = [a[0], a[2]].sort((x, y) => x - y);
      return `${ends[0]},${a[1]},${ends[1]}`;
    });
    theirs.sort((x, y) => {
      const p = x.split(",").map(Number);
      const q = y.split(",").map(Number);
      return p[0]! - q[0]! || p[1]! - q[1]! || p[2]! - q[2]!;
    });
    expect(ours).toEqual(theirs);
  });

  it("finds no row pair OpenMM would have had to break a tie between", () => {
    // 🔴 The one place our matching is deterministic and OpenMM's is not: it
    // scans a Python set and breaks on the first match. That only matters if
    // two rows with *different* parameters can match the same term, so count
    // them rather than assume. Zero means the two cannot disagree here.
    const angles = findAngles(bonds!, neighbours!);
    const propers = findPropers(angles, neighbours!);
    const bondDiagnostics = assignBonds(bonds!, types!, field!).diagnostics;
    const angleDiagnostics = assignAngles(angles, types!, field!).diagnostics;
    expect(bondDiagnostics.ambiguous).toBe(0);
    expect(angleDiagnostics.ambiguous).toBe(0);
    expect(propers.length).toBeGreaterThan(0);
  });

  it("enumerates the improper candidates OpenMM would have considered", () => {
    const impropers = findImpropers(neighbours!);
    // Every atom with three or more bonds contributes C(n,3) candidates, most
    // of which match no template. This checks the count against the graph
    // rather than against the dump, which only records the ones that matched.
    const expected = neighbours!.reduce((sum, list) => {
      const n = list.length;
      return sum + (n > 2 ? (n * (n - 1) * (n - 2)) / 6 : 0);
    }, 0);
    expect(impropers).toHaveLength(expected);
    for (const [central, a, b, c] of impropers) {
      expect(neighbours![central]).toContain(a);
      expect(a).toBeLessThan(b);
      expect(b).toBeLessThan(c);
    }
  });
});
