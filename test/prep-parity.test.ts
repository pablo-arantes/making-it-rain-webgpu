/**
 * Phase 3's gate: term-by-term parameter parity, on every fixture we have.
 *
 * The per-piece tests check one thing each against 1L2Y and say *why* each
 * rule is what it is. This file checks the whole pipeline against every
 * fixture, and exists for a different reason: a rule can be right for 1L2Y and
 * wrong in general, and nothing in a single-fixture test can tell.
 *
 * 1L2Y is a 20-residue peptide with no histidine, no cysteine and no sulfur.
 * 3HTB's receptor is T4 lysozyme -- 163 residues, 2614 atoms, all twenty
 * standard residues. It adds the cases 1L2Y cannot reach:
 *
 *   - **HID vs HIE**, which share an element signature and are separated only
 *     by which ring nitrogen carries the proton. That is the graph matcher
 *     deciding a protonation state, and the charges differ if it decides wrong.
 *   - **methionine's sulfur** and two cysteines, so the element handling is not
 *     only ever C/N/O/H.
 *   - 8842 torsions and 14232 exceptions, which is enough for a rule that is
 *     wrong in a rare case to actually meet that case.
 *
 * Everything is compared against `tools/dump_parameters.py` and
 * `tools/dump_template_match.py`. Fixtures are regenerated rather than
 * committed, so a missing one skips and says so rather than passing silently.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { parseStandardBonds } from "../src/prep/bonds.js";
import { parseForceField } from "../src/prep/ffxml.js";
import { prepareSystem } from "../src/prep/prepare.js";

interface Dump {
  readonly atoms: number;
  readonly masses: readonly number[];
  readonly topology: { readonly bonds: readonly (readonly [number, number])[] };
  readonly bonds: readonly (readonly [number, number, number, number])[];
  readonly angles: readonly (readonly [number, number, number, number, number])[];
  readonly torsions: readonly (readonly number[])[];
  readonly nonbonded: {
    readonly particles: readonly (readonly [number, number, number])[];
    readonly exceptions: readonly (readonly [number, number, number, number, number])[];
  };
}

interface Capture {
  readonly residues: readonly { readonly template: string }[];
  readonly templateIndexes: readonly number[];
  readonly types: readonly string[];
  readonly bondedToAtom: readonly (readonly number[])[];
}

const fieldPath = "vendor/ffxml/protein.ff14SB.xml";
const residuesPath = "vendor/ffxml/residues.xml";

/** The fixtures, with what each one is here to cover. */
const FIXTURES = [
  { name: "1L2Y", atoms: 304, note: "a 20-residue peptide, both chain termini" },
  { name: "3HTB", atoms: 2614, note: "T4 lysozyme, all twenty residues, HID and CYS" },
] as const;

for (const fixture of FIXTURES) {
  const pdbPath = `fixtures/prep/${fixture.name}.pdb`;
  const dumpPath = `fixtures/prep/${fixture.name}-params.json`;
  const matchPath = `fixtures/prep/${fixture.name}-match.json`;
  const present =
    existsSync(pdbPath) &&
    existsSync(dumpPath) &&
    existsSync(matchPath) &&
    existsSync(fieldPath);

  describe.skipIf(!present)(`parameter parity: ${fixture.name} (${fixture.note})`, () => {
    const prepared = present
      ? prepareSystem(
          readPdb(readFileSync(pdbPath, "utf8")),
          parseForceField(readFileSync(fieldPath, "utf8")),
          parseStandardBonds(readFileSync(residuesPath, "utf8")),
        )
      : undefined;
    const dump = present
      ? (JSON.parse(readFileSync(dumpPath, "utf8")) as Dump)
      : undefined;
    const capture = present
      ? (JSON.parse(readFileSync(matchPath, "utf8")) as Capture)
      : undefined;

    it("reads the expected structure", () => {
      expect(prepared!.topology.atoms).toHaveLength(fixture.atoms);
      expect(prepared!.topology.atoms).toHaveLength(dump!.atoms);
    });

    it("reconstructs OpenMM's bonds", () => {
      const ours = new Set(
        prepared!.bonds.map(([a, b]) => `${Math.min(a, b)},${Math.max(a, b)}`),
      );
      const theirs = new Set(
        dump!.topology.bonds.map(([a, b]) => `${Math.min(a, b)},${Math.max(a, b)}`),
      );
      expect([...ours].sort()).toEqual([...theirs].sort());
    });

    it("chooses OpenMM's templates and atom correspondence", () => {
      expect(prepared!.assignment.atoms.map((a) => a.type)).toEqual([
        ...capture!.types,
      ]);
      expect(prepared!.assignment.matches.map((m) => m.template)).toEqual(
        capture!.residues.map((r) => r.template),
      );
      const indexes = new Array<number>(prepared!.topology.atoms.length);
      for (const match of prepared!.assignment.matches) {
        const residue = prepared!.topology.residues[match.residue]!;
        residue.atoms.forEach((atom, position) => {
          indexes[atom] = match.atomMatches[position]!;
        });
      }
      expect(indexes).toEqual([...capture!.templateIndexes]);
    });

    it("assigns OpenMM's masses, charges and Lennard-Jones parameters", () => {
      const atoms = prepared!.assignment.atoms;
      expect(atoms.map((a) => a.mass)).toEqual([...dump!.masses]);
      atoms.forEach((atom, index) => {
        const [charge, sigma, epsilon] = dump!.nonbonded.particles[index]!;
        expect(atom.charge).toBeCloseTo(charge, 12);
        expect(atom.sigma).toBeCloseTo(sigma, 10);
        expect(atom.epsilon).toBeCloseTo(epsilon, 10);
      });
    });

    it("leaves no term unmatched and no row ambiguous", () => {
      const { bonds, angles, propers } = prepared!.diagnostics;
      expect({
        bonds: bonds.unmatched,
        angles: angles.unmatched,
        propers: propers.unmatched,
      }).toEqual({ bonds: 0, angles: 0, propers: 0 });
      expect({
        bonds: bonds.ambiguous,
        angles: angles.ambiguous,
        propers: propers.ambiguous,
      }).toEqual({ bonds: 0, angles: 0, propers: 0 });
    });

    it("reproduces every bond and angle term", () => {
      expect(prepared!.bondTerms).toHaveLength(dump!.bonds.length);
      const bondRows = new Map(
        dump!.bonds.map((b) => [`${Math.min(b[0], b[1])},${Math.max(b[0], b[1])}`, b]),
      );
      for (const term of prepared!.bondTerms) {
        const [a, b] = term.atoms as [number, number];
        const row = bondRows.get(`${Math.min(a, b)},${Math.max(a, b)}`)!;
        expect(row, `no OpenMM bond for ${a}-${b}`).toBeDefined();
        expect(term.value).toBeCloseTo(row[2], 12);
        expect(term.k).toBeCloseTo(row[3], 7);
      }

      expect(prepared!.angleTerms).toHaveLength(dump!.angles.length);
      const angleRows = new Map(
        dump!.angles.map((a) => {
          const ends = [a[0], a[2]].sort((x, y) => x - y);
          return [`${ends[0]},${a[1]},${ends[1]}`, a];
        }),
      );
      for (const term of prepared!.angleTerms) {
        const [a, b, c] = term.atoms as [number, number, number];
        const ends = [a, c].sort((x, y) => x - y);
        const row = angleRows.get(`${ends[0]},${b},${ends[1]}`)!;
        expect(row, `no OpenMM angle for ${a}-${b}-${c}`).toBeDefined();
        expect(term.value).toBeCloseTo(row[3], 12);
        expect(term.k).toBeCloseTo(row[4], 7);
      }
    });

    it("reproduces every torsion, atom order included", () => {
      const key = (
        atoms: readonly number[],
        periodicity: number,
        phase: number,
        k: number,
      ): string => `${atoms.join(",")}|${periodicity}|${phase.toFixed(12)}|${k.toFixed(9)}`;
      const count = (keys: readonly string[]): Map<string, number> => {
        const counts = new Map<string, number>();
        for (const entry of keys) counts.set(entry, (counts.get(entry) ?? 0) + 1);
        return counts;
      };
      const mine = count(
        prepared!.torsionTerms.map((t) => key(t.atoms, t.periodicity, t.phase, t.k)),
      );
      const theirs = count(
        dump!.torsions.map((t) => key(t.slice(0, 4), t[4]!, t[5]!, t[6]!)),
      );
      expect(prepared!.torsionTerms).toHaveLength(dump!.torsions.length);

      const missing: string[] = [];
      for (const [entry, n] of theirs) {
        if ((mine.get(entry) ?? 0) < n) missing.push(entry);
      }
      const extra: string[] = [];
      for (const [entry, n] of mine) {
        if ((theirs.get(entry) ?? 0) < n) extra.push(entry);
      }
      expect({ missing, extra }).toEqual({ missing: [], extra: [] });
    });

    it("reproduces every nonbonded exception", () => {
      expect(prepared!.exceptions).toHaveLength(dump!.nonbonded.exceptions.length);
      const rows = new Map(
        dump!.nonbonded.exceptions.map((e) => [
          `${Math.min(e[0], e[1])},${Math.max(e[0], e[1])}`,
          e,
        ]),
      );
      for (const exception of prepared!.exceptions) {
        const [a, b] = exception.atoms;
        const row = rows.get(`${Math.min(a, b)},${Math.max(a, b)}`)!;
        expect(row, `no OpenMM exception for ${a}-${b}`).toBeDefined();
        expect(exception.chargeProd).toBeCloseTo(row[2], 12);
        expect(exception.sigma).toBeCloseTo(row[3], 12);
        expect(exception.epsilon).toBeCloseTo(row[4], 12);
      }
    });
  });
}
