/**
 * Improper torsions, and the ordering rules Amber needs.
 *
 * Impropers are the fiddliest part of parameter assignment, and the reason is
 * that Amber defines nearly all of them with wildcards -- `X-X-C-O` -- which
 * leaves the atom order ambiguous. A periodic torsion's energy depends on the
 * dihedral angle, and that angle changes sign if you swap two of the atoms, so
 * "ambiguous" is not "harmless": picking a different order from OpenMM gives a
 * different energy for the same structure. `ordering="amber"` in the ffxml
 * selects the tie-breaking rules, and this file transcribes them from
 * `_matchImproper` in OpenMM's `forcefield.py`.
 *
 * Four things that are not guessable and all come from reading that function:
 *
 * **The central atom ends up third.** The match is `(a2, a3, torsion[0], a4)`,
 * not `(central, ...)`. The candidate enumeration puts the central atom first,
 * so the order changes between finding and emitting.
 *
 * **The last non-wildcard match wins, not the first.** Unlike proper torsions,
 * which break out of the scan on the first specific match, the improper scan
 * keeps going and lets a later specific row overwrite an earlier one. It only
 * skips *wildcard* rows once something has matched. So the rule is: the last
 * specific match, or failing that the first wildcard match.
 *
 * **Which tie-break applies depends on whether the row has a wildcard.** With
 * no wildcard the swaps compare **type names**; with a wildcard they compare
 * **elements**. Same code shape, different quantity, and using one for the
 * other is invisible on most atoms.
 *
 * **Ties are broken by residue index then template atom index** -- the atom's
 * position inside the residue template it matched, not its index in the file.
 * That is why `matchResidues` returns `atomMatches`: this is the only thing
 * that needs it, and it cannot be recovered afterwards.
 *
 * The first permutation of the three outer atoms that fits the row's slots is
 * the one used, so the permutation order has to be Python's
 * `itertools.permutations` order exactly.
 */

import type { ForceField, TorsionRow } from "./ffxml.js";
import { resolveSlots, type Quad, type TorsionTerm } from "./bonded.js";

/** What the ordering rules need to know about each atom. */
export interface AtomContext {
  readonly type: string;
  readonly element: string;
  /** amu, used by the `default` ordering's mass comparison. */
  readonly mass: number;
  readonly residue: number;
  /** The atom's index within the residue template it matched. */
  readonly templateIndex: number;
}

type Slot = ReadonlySet<string> | null;

function accepts(slot: Slot | undefined, type: string): boolean {
  const resolved = slot ?? null;
  return resolved === null || resolved.has(type);
}

/**
 * Python's `itertools.permutations` order for three items, as position triples.
 *
 * Written out rather than generated, because the order is load-bearing: the
 * first fitting permutation is the one used, so a different enumeration order
 * silently changes which atoms the torsion is built from.
 */
const PERMUTATIONS: readonly (readonly [number, number, number])[] = [
  [1, 2, 3],
  [1, 3, 2],
  [2, 1, 3],
  [2, 3, 1],
  [3, 1, 2],
  [3, 2, 1],
];

interface Resolved {
  readonly row: TorsionRow;
  readonly slots: readonly Slot[];
  readonly wildcard: boolean;
}

export function resolveImpropers(field: ForceField): readonly Resolved[] {
  return field.impropers.map((row) => {
    const slots = resolveSlots(row, field);
    return { row, slots, wildcard: slots.some((slot) => slot === null) };
  });
}

/** `(a1, a2, a3, a4)` in emission order, plus the row they came from. */
interface ImproperMatch {
  readonly atoms: Quad;
  readonly row: TorsionRow;
}

/**
 * Order the three outer atoms the way `ordering="amber"` says to.
 *
 * Transcribed branch for branch. The comparisons are `>` on the pair
 * `(residue, templateIndex)`, applied in a fixed sequence of three
 * conditional swaps -- not a sort, and the sequence matters because each swap
 * sees the result of the previous one.
 */
function amberOrder(
  a2: number,
  a3: number,
  a4: number,
  keys: readonly [string, string, string],
  wildcard: boolean,
  context: readonly AtomContext[],
): readonly [number, number, number] {
  let first = a2;
  let second = a3;
  let third = a4;
  const after = (x: number, y: number): boolean => {
    const p = context[x] as AtomContext;
    const q = context[y] as AtomContext;
    return p.residue > q.residue || (p.residue === q.residue && p.templateIndex > q.templateIndex);
  };

  if (!wildcard) {
    // No wildcard: the slots named types, so equality of *type* is what makes
    // two positions interchangeable.
    if (keys[0] === keys[2] && after(first, third)) {
      [first, third] = [third, first];
    }
    if (keys[1] === keys[2] && after(second, third)) {
      [second, third] = [third, second];
    }
    if (keys[0] === keys[1] && after(first, second)) {
      [first, second] = [second, first];
    }
  } else {
    // Wildcard: the slots accepted anything, so equality of *element* is the
    // test instead. Different quantity, same shape.
    //
    // 🔴 The elements are read **once**, from the atoms as the permutation
    // delivered them, and are not re-read after a swap -- while the residue
    // and template indices are. That asymmetry is in OpenMM's code and looks
    // accidental. It makes no difference on 1L2Y (measured: re-reading them
    // changes none of the 1017 torsions), so this is transcribed rather than
    // justified: a molecule where the two differ would be one where our
    // energy differs from OpenMM's, and matching is the point.
    const e2 = (context[first] as AtomContext).element;
    const e3 = (context[second] as AtomContext).element;
    const e4 = (context[third] as AtomContext).element;
    if (e2 === e4 && after(first, third)) {
      [first, third] = [third, first];
    }
    if (e3 === e4 && after(second, third)) {
      [second, third] = [third, second];
    }
    if (after(first, second)) {
      [first, second] = [second, first];
    }
  }
  return [first, second, third];
}

/** The `default` ordering's rules, which Amber's own files do not use. */
function defaultOrder(
  a1: number,
  a2: number,
  context: readonly AtomContext[],
): readonly [number, number] {
  const first = context[a1] as AtomContext;
  const second = context[a2] as AtomContext;
  if (first.element === second.element && a1 > a2) return [a2, a1];
  if (
    first.element !== "C" &&
    (second.element === "C" || first.mass < second.mass)
  ) {
    return [a2, a1];
  }
  return [a1, a2];
}

/** Match one improper candidate, or `null` if no row applies. */
export function matchImproper(
  candidate: Quad,
  context: readonly AtomContext[],
  resolved: readonly Resolved[],
): ImproperMatch | null {
  const types = candidate.map((atom) => (context[atom] as AtomContext).type) as [
    string,
    string,
    string,
    string,
  ];
  let match: ImproperMatch | null = null;

  for (const { row, slots, wildcard } of resolved) {
    // Once anything has matched, wildcard rows are skipped -- but specific
    // rows are still tried, and overwrite. See the module comment.
    if (match !== null && wildcard) continue;
    if (!accepts(slots[0], types[0])) continue;

    for (const permutation of PERMUTATIONS) {
      const keys = permutation.map((position) => types[position] as string) as [
        string,
        string,
        string,
      ];
      if (
        !accepts(slots[1], keys[0]) ||
        !accepts(slots[2], keys[1]) ||
        !accepts(slots[3], keys[2])
      ) {
        continue;
      }
      const atoms = permutation.map((position) => candidate[position] as number) as [
        number,
        number,
        number,
      ];
      if (row.ordering === "amber") {
        const [first, second, third] = amberOrder(
          atoms[0],
          atoms[1],
          atoms[2],
          keys,
          wildcard,
          context,
        );
        // 🔴 The central atom goes third.
        match = { atoms: [first, second, candidate[0], third], row };
      } else if (row.ordering === "charmm" && !wildcard) {
        match = { atoms: [candidate[0], atoms[0], atoms[1], atoms[2]], row };
      } else if (row.ordering === "smirnoff") {
        match = { atoms: [candidate[0], atoms[0], atoms[1], atoms[2]], row };
      } else {
        // `default`, and `charmm` with a wildcard, share these rules.
        const [first, second] = defaultOrder(atoms[0], atoms[1], context);
        match = { atoms: [first, second, candidate[0], atoms[2]], row };
      }
      break;
    }
  }
  return match;
}

/**
 * Assign impropers, with OpenMM's match cache.
 *
 * 🔴 The cache is not an optimisation we could skip -- it changes the answer,
 * and it took a term-by-term comparison to find.
 *
 * OpenMM keys `impr_cache` by the candidate's **type signature** and stores the
 * *positions* its ordering rules chose, then replays those positions on every
 * later candidate with the same signature. But the rules consume residue and
 * template indices, which differ between candidates. So the rules effectively
 * run once per signature and the first candidate to carry it decides the atom
 * order for all the rest.
 *
 * Measured on 1L2Y: two of the 1017 torsions differ without this. Tyrosine's
 * ring gives candidate `(49, 45, 50, 53)` the signature
 * `(CA, CA, HA, C)` and positions `(1, 3, 0, 2)`; the later candidate
 * `(51, 47, 52, 53)` would freshly compute `(3, 1, 0, 2)` but replays the
 * cached positions instead, giving `(47, 53, 51, 52)` rather than
 * `(53, 47, 51, 52)`. Swapping a torsion's first two atoms flips the sign of
 * its dihedral angle, so this is an energy difference, not a bookkeeping one.
 *
 * It also means the candidate *order* matters, which is why `neighbourLists`
 * has to agree with OpenMM's `bondedToAtom` -- `test/prep-impropers.test.ts`
 * asserts that against a capture rather than trusting it.
 */
export function assignImpropers(
  candidates: readonly Quad[],
  context: readonly AtomContext[],
  field: ForceField,
): readonly TorsionTerm[] {
  const resolved = resolveImpropers(field);
  const terms: TorsionTerm[] = [];
  // `null` records a signature already known not to match, which OpenMM stores
  // as -1 and checks before anything else.
  const cache = new Map<
    string,
    { positions: readonly [number, number, number, number]; row: TorsionRow } | null
  >();

  for (const candidate of candidates) {
    const signature = candidate
      .map((atom) => (context[atom] as AtomContext).type)
      .join("\u0000");

    let match: ImproperMatch | null;
    if (cache.has(signature)) {
      const cached = cache.get(signature) ?? null;
      if (cached === null) continue;
      match = {
        atoms: cached.positions.map((position) => candidate[position] as number) as unknown as Quad,
        row: cached.row,
      };
    } else {
      match = matchImproper(candidate, context, resolved);
      if (match === null) {
        cache.set(signature, null);
        continue;
      }
      const positions = match.atoms.map((atom) =>
        candidate.indexOf(atom),
      ) as unknown as readonly [number, number, number, number];
      cache.set(signature, { positions, row: match.row });
    }
    if (match === null) continue;
    for (const term of match.row.terms) {
      if (term.k === 0) continue;
      if (match.row.ordering === "smirnoff") {
        // A trefoil: all three cyclic orders get the same parameters.
        const [a1, a2, a3, a4] = match.atoms;
        for (const quad of [
          [a1, a2, a3, a4],
          [a1, a3, a4, a2],
          [a1, a4, a2, a3],
        ] as Quad[]) {
          terms.push({
            atoms: quad,
            periodicity: term.periodicity,
            phase: term.phase,
            k: term.k,
          });
        }
        continue;
      }
      terms.push({
        atoms: match.atoms,
        periodicity: term.periodicity,
        phase: term.phase,
        k: term.k,
      });
    }
  }
  return terms;
}
