/**
 * Improper torsions, the amber ordering rules, and the match cache.
 *
 * All 1017 torsions OpenMM created for 1L2Y, compared **with their atom order
 * intact**. That is the point of this file: a periodic torsion's energy depends
 * on the dihedral angle, and swapping two of the four atoms flips that angle's
 * sign. So an improper with the right parameters and the wrong atom order is a
 * wrong energy, and a comparison that canonicalised the order would not see it.
 *
 * Two of the three bugs this file exists to prevent were found by exactly this
 * comparison, both in the last two torsions of the 1017:
 *
 *   - the ordering rules run against `atomTemplateIndexes`, the atom's position
 *     inside its matched template, which only the matcher knows;
 *   - OpenMM caches improper matches by type signature and replays the
 *     *positions*, so the ordering rules effectively run once per signature and
 *     the first candidate to carry it decides the order for every later one.
 *
 * The second is tested directly below, by computing the uncached answer and
 * checking it really does differ -- so the cache cannot be removed as dead
 * weight by someone who has not read this.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import { assignAtoms, atomContexts } from "../src/prep/assign.js";
import {
  assignPropers,
  findAngles,
  findImpropers,
  findPropers,
  neighbourLists,
} from "../src/prep/bonded.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";
import { parseForceField } from "../src/prep/ffxml.js";
import {
  assignImpropers,
  matchImproper,
  resolveImpropers,
} from "../src/prep/torsions.js";

interface Dump {
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

interface Match {
  readonly residues: readonly { readonly index: number; readonly template: string }[];
  readonly templateIndexes: readonly number[];
  readonly types: readonly string[];
  readonly bondedToAtom: readonly (readonly number[])[];
}

const pdbPath = "fixtures/prep/1L2Y.pdb";
const dumpPath = "fixtures/prep/1L2Y-params.json";
const matchPath = "fixtures/prep/1L2Y-match.json";
const fieldPath = "vendor/ffxml/protein.ff14SB.xml";
const present =
  existsSync(pdbPath) &&
  existsSync(dumpPath) &&
  existsSync(matchPath) &&
  existsSync(fieldPath);

describe.skipIf(!present)("improper torsions for 1L2Y", () => {
  const topology = present ? readPdb(readFileSync(pdbPath, "utf8")) : undefined;
  const dump = present
    ? (JSON.parse(readFileSync(dumpPath, "utf8")) as Dump)
    : undefined;
  const capture = present
    ? (JSON.parse(readFileSync(matchPath, "utf8")) as Match)
    : undefined;
  const field = present ? parseForceField(readFileSync(fieldPath, "utf8")) : undefined;
  const bonds = present
    ? inferBonds(
        topology!,
        parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
      )
    : undefined;
  const assignment = present ? assignAtoms(topology!, bonds!, field!) : undefined;
  const context = present ? atomContexts(topology!, assignment!) : undefined;
  const neighbours = present
    ? neighbourLists(topology!.atoms.length, bonds!)
    : undefined;

  /** Every torsion as an exact key, atom order included. */
  const torsionKey = (
    atoms: readonly number[],
    periodicity: number,
    phase: number,
    k: number,
  ): string => `${atoms.join(",")}|${periodicity}|${phase.toFixed(12)}|${k.toFixed(10)}`;

  it("matches OpenMM's template choice and atom correspondence exactly", () => {
    // 🔴 A direct comparison, not an inference. The per-atom charges in
    // test/prep-assign.test.ts cannot distinguish two valid isomorphisms of a
    // symmetric ring -- tyrosine's CD1/CD2 carry identical charges -- but the
    // improper ordering rules consume the template index, so the choice is
    // observable in the energy. This pins it.
    expect(assignment!.atoms.map((a) => a.type)).toEqual([...capture!.types]);
    const ours = assignment!.matches.flatMap((match) => {
      const residue = topology!.residues[match.residue]!;
      return residue.atoms.map((atom, position) => [atom, match.atomMatches[position]!]);
    });
    const indexes = new Array<number>(topology!.atoms.length);
    for (const [atom, index] of ours) indexes[atom!] = index!;
    expect(indexes).toEqual([...capture!.templateIndexes]);
    expect(assignment!.matches.map((m) => m.template)).toEqual(
      capture!.residues.map((r) => r.template),
    );
  });

  it("builds the neighbour lists OpenMM built, in the same order", () => {
    // 🔴 Order, not just content. Improper candidates are combinations of this
    // list and the match cache is keyed by type signature, so which candidate
    // comes first changes the atom order of every later one that shares a
    // signature. We sort; OpenMM appends in bond order. Those agree for 1L2Y
    // -- all 304 lists are already sorted -- and this asserts it rather than
    // assuming it, so a fixture where they diverge fails here and names the
    // reason instead of shifting an energy.
    expect(neighbours!.map((list) => [...list])).toEqual(
      capture!.bondedToAtom.map((list) => [...list]),
    );
  });

  it("reproduces all 1017 torsions, atom order included", () => {
    const angles = findAngles(bonds!, neighbours!);
    const propers = findPropers(angles, neighbours!);
    const impropers = findImpropers(neighbours!);
    const types = assignment!.atoms.map((a) => a.type);

    const ours = [
      ...assignPropers(propers, types, field!).terms,
      ...assignImpropers(impropers, context!, field!),
    ];
    expect(ours).toHaveLength(dump!.torsions.length);

    const count = (keys: readonly string[]): Map<string, number> => {
      const counts = new Map<string, number>();
      for (const key of keys) counts.set(key, (counts.get(key) ?? 0) + 1);
      return counts;
    };
    const mine = count(ours.map((t) => torsionKey(t.atoms, t.periodicity, t.phase, t.k)));
    const theirs = count(
      dump!.torsions.map((t) => torsionKey([t[0], t[1], t[2], t[3]], t[4], t[5], t[6])),
    );

    const missing: string[] = [];
    for (const [key, n] of theirs) {
      if ((mine.get(key) ?? 0) < n) missing.push(key);
    }
    const extra: string[] = [];
    for (const [key, n] of mine) {
      if ((theirs.get(key) ?? 0) < n) extra.push(key);
    }
    expect({ missing, extra }).toEqual({ missing: [], extra: [] });
  });

  it("needs the match cache: without it, two torsions come out reordered", () => {
    // The negative control for the cache. Computing every candidate freshly is
    // what a reasonable implementation would do, and it is wrong for exactly
    // two of 1L2Y's torsions. If this ever reports 0, the cache has stopped
    // mattering and the claim in src/prep/torsions.ts needs re-measuring.
    const impropers = findImpropers(neighbours!);
    const resolved = resolveImpropers(field!);
    const cached = assignImpropers(impropers, context!, field!);

    const fresh: string[] = [];
    for (const candidate of impropers) {
      const match = matchImproper(candidate, context!, resolved);
      if (match === null) continue;
      for (const term of match.row.terms) {
        if (term.k === 0) continue;
        fresh.push(torsionKey(match.atoms, term.periodicity, term.phase, term.k));
      }
    }
    const cachedKeys = cached.map((t) =>
      torsionKey(t.atoms, t.periodicity, t.phase, t.k),
    );
    expect(fresh).toHaveLength(cachedKeys.length);
    const differing = fresh.filter((key, index) => key !== cachedKeys[index]);
    expect(differing).toHaveLength(2);
  });
});
