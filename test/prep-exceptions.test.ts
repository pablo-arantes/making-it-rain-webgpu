/**
 * Nonbonded exceptions for 1L2Y, against OpenMM's own.
 *
 * 1687 of them: 875 removed outright and 812 scaled 1-4 pairs. Compared on
 * every field, plus the order they come out in, because the engine contract is
 * a serialised `System` and the exception order is part of that file.
 *
 * The 1-4 count is the part worth having a number for. Scaling a pair that
 * should be excluded, or excluding one that should be scaled, moves the energy
 * by a lot and is invisible in a count of exceptions -- both mistakes give
 * 1687. So the split is asserted as well as the total.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { assignAtoms } from "../src/prep/assign.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";
import { findExceptions, withinBonds } from "../src/prep/exceptions.js";
import { parseForceField } from "../src/prep/ffxml.js";

interface Dump {
  readonly nonbonded: {
    readonly exceptions: readonly (readonly [number, number, number, number, number])[];
  };
}

const pdbPath = "fixtures/prep/1L2Y.pdb";
const dumpPath = "fixtures/prep/1L2Y-params.json";
const fieldPath = "vendor/ffxml/protein.ff14SB.xml";
const present = existsSync(pdbPath) && existsSync(dumpPath) && existsSync(fieldPath);

describe.skipIf(!present)("nonbonded exceptions for 1L2Y", () => {
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
  const atoms = present ? assignAtoms(topology!, bonds!, field!).atoms : undefined;
  const exceptions = present
    ? findExceptions(atoms!, bonds!, field!.coulomb14Scale, field!.lj14Scale)
    : undefined;

  it("reads Amber's 1-4 scale factors", () => {
    expect(field!.coulomb14Scale).toBeCloseTo(5 / 6, 15);
    expect(field!.lj14Scale).toBe(0.5);
  });

  it("creates every exception OpenMM created, with the same parameters", () => {
    expect(exceptions).toHaveLength(dump!.nonbonded.exceptions.length);
    const expected = new Map(
      dump!.nonbonded.exceptions.map((e) => [
        `${Math.min(e[0], e[1])},${Math.max(e[0], e[1])}`,
        e,
      ]),
    );
    for (const exception of exceptions!) {
      const [a, b] = exception.atoms;
      const row = expected.get(`${Math.min(a, b)},${Math.max(a, b)}`);
      expect(row, `no OpenMM exception for ${a}-${b}`).toBeDefined();
      expect(exception.chargeProd).toBeCloseTo(row![2], 12);
      expect(exception.sigma).toBeCloseTo(row![3], 12);
      expect(exception.epsilon).toBeCloseTo(row![4], 12);
    }
  });

  it("splits them 875 excluded and 812 scaled", () => {
    const scaled = exceptions!.filter((e) => e.epsilon !== 0 || e.chargeProd !== 0);
    expect(scaled).toHaveLength(812);
    expect(exceptions!.length - scaled.length).toBe(875);
    // Every removed one carries the placeholder sigma, not the pair's mean.
    for (const exception of exceptions!) {
      if (exception.epsilon === 0 && exception.chargeProd === 0) {
        expect(exception.sigma).toBe(1);
      }
    }
  });

  it("emits them in OpenMM's order: ascending pairs, sorted by the second atom", () => {
    // Measured from the built System, not assumed -- see src/prep/exceptions.ts.
    const pairs = exceptions!.map((e) => e.atoms);
    for (const [a, b] of pairs) expect(a).toBeLessThan(b);
    const sorted = [...pairs].sort((p, q) => p[1] - q[1] || p[0] - q[0]);
    expect(pairs).toEqual(sorted);
  });

  it("treats a 1-4 as distance three, not merely 'has a three-bond path'", () => {
    // 🔴 The two differ only in a ring, where a pair three bonds apart one way
    // round is two bonds apart the other. A test of "is there a three-bond
    // path" would scale those pairs instead of excluding them, which is a real
    // energy error on every tryptophan and proline.
    //
    // Measured on 1L2Y: 25 pairs have a simple three-bond path *and* sit
    // within two bonds, all of them in tryptophan's fused five-membered ring
    // (CG-NE1, CG-CE2, CD1-CD2, CD1-CE2, CD2-NE1 among them) and in the four
    // prolines' rings. All 25 must come out excluded, and the count is
    // asserted so that a fixture without a five-membered ring cannot quietly
    // stop covering this.
    const adjacency: number[][] = atoms!.map(() => []);
    for (const [a, b] of bonds!) {
      adjacency[a]!.push(b);
      adjacency[b]!.push(a);
    }
    const within2 = withinBonds(atoms!.length, bonds!, 2);
    const shortcut = new Set<string>();
    for (let i = 0; i < atoms!.length; i++) {
      for (const x of adjacency[i]!) {
        for (const y of adjacency[x]!) {
          if (y === i) continue;
          for (const j of adjacency[y]!) {
            if (j === x || j === i || j <= i) continue;
            if (within2[i]!.has(j)) shortcut.add(`${i},${j}`);
          }
        }
      }
    }
    expect(shortcut.size).toBe(25);

    const byPair = new Map(
      exceptions!.map((e) => [`${e.atoms[0]},${e.atoms[1]}`, e]),
    );
    for (const key of shortcut) {
      const exception = byPair.get(key);
      expect(exception, `no exception for the ring pair ${key}`).toBeDefined();
      expect(exception!.epsilon).toBe(0);
      expect(exception!.chargeProd).toBe(0);
    }
  });
});
