/**
 * Enumerating bonds, angles and torsions, and matching them to parameters.
 *
 * Two halves, kept separate because they fail differently. The enumeration is
 * pure graph work and is transcribed from `ForceField.createSystem`: a wrong
 * canonical ordering there drops or doubles terms. The matching is table
 * lookup, transcribed from `HarmonicBondGenerator.createForce`,
 * `HarmonicAngleGenerator.createForce` and `PeriodicTorsionGenerator`, and a
 * wrong key there gives a term the wrong stiffness.
 *
 * Three rules that are easy to get wrong and are all measured in
 * `test/prep-bonded.test.ts`:
 *
 * **A slot is a set of types, not a type.** `class1="CT"` means every type in
 * class CT; `type1="CT"` means that one type. `parseForceField` keeps the two
 * apart (`keys` plus `byClass`) and this file expands them, because collapsing
 * them in the parser would make a class-keyed row silently match one type.
 *
 * **An empty class is a wildcard, and specificity is a preference, not a
 * filter.** Amber writes most impropers and many propers as `X-C-CA-X`. OpenMM
 * keeps scanning after a wildcard match and takes a non-wildcard one if it
 * finds it, so a wildcard row is a fallback. Treating the first match as final
 * picks the wrong torsion for anything Amber bothered to make specific.
 *
 * **Matching is symmetric, in a specific way.** A bond matches either way
 * round. An angle matches either way round with the *middle slot fixed*. A
 * torsion matches forwards or fully reversed -- not any permutation.
 *
 * 🔴 One place OpenMM is order-dependent and we are not. It scans
 * `bondsForAtomType[type1]`, which is a **set**, and breaks on the first match;
 * if two rows with different parameters both matched, which one wins is
 * whatever Python's set iteration gives. We scan in file order instead, which
 * is deterministic. That is a difference only if such a row pair exists, so the
 * test counts them rather than assuming: 0 for ff14SB on 1L2Y, so the two
 * agree. If a force field ever produces one, the count fails the test instead
 * of quietly diverging.
 */

import type { ForceField, ParameterRow, TorsionRow } from "./ffxml.js";

export type Quad = readonly [number, number, number, number];
export type Triple = readonly [number, number, number];
export type Pair = readonly [number, number];

/** A resolved slot: the set of types it accepts, or `null` for a wildcard. */
type Slot = ReadonlySet<string> | null;

export interface HarmonicTerm {
  readonly atoms: readonly number[];
  /** nm for a bond, radians for an angle. */
  readonly value: number;
  /** kJ/mol/nm^2 for a bond, kJ/mol/rad^2 for an angle. */
  readonly k: number;
}

export interface TorsionTerm {
  readonly atoms: Quad;
  readonly periodicity: number;
  /** radians. */
  readonly phase: number;
  /** kJ/mol. */
  readonly k: number;
}

/** How many terms matched more than one row at the winning specificity. */
export interface MatchDiagnostics {
  ambiguous: number;
  unmatched: number;
}

/** Adjacency, which every enumeration below walks. */
export function neighbourLists(
  atomCount: number,
  bonds: readonly Pair[],
): readonly (readonly number[])[] {
  const neighbours: number[][] = Array.from({ length: atomCount }, () => []);
  for (const [a, b] of bonds) {
    (neighbours[a] as number[]).push(b);
    (neighbours[b] as number[]).push(a);
  }
  // Sorted, because `findImpropers` takes combinations in neighbour order and
  // OpenMM's `bondedToAtom` is built by appending in bond order. Sorting makes
  // our enumeration independent of the order the bond list arrived in, which
  // the bond reconstruction does not promise.
  for (const list of neighbours) list.sort((a, b) => a - b);
  return neighbours;
}

/**
 * Every A-B-C where A-B and B-C are bonded, canonicalised so the lower end
 * comes first, then sorted. Transcribed from `createSystem`'s `uniqueAngles`.
 */
export function findAngles(
  bonds: readonly Pair[],
  neighbours: readonly (readonly number[])[],
): readonly Triple[] {
  const unique = new Set<string>();
  const add = (a: number, b: number, c: number): void => {
    unique.add(a < c ? `${a},${b},${c}` : `${c},${b},${a}`);
  };
  for (const [atom1, atom2] of bonds) {
    for (const atom of neighbours[atom1] as readonly number[]) {
      if (atom !== atom2) add(atom, atom1, atom2);
    }
    for (const atom of neighbours[atom2] as readonly number[]) {
      if (atom !== atom1) add(atom1, atom2, atom);
    }
  }
  return sortTuples(unique) as readonly Triple[];
}

/** Every A-B-C-D along three consecutive bonds. `uniquePropers`. */
export function findPropers(
  angles: readonly Triple[],
  neighbours: readonly (readonly number[])[],
): readonly Quad[] {
  const unique = new Set<string>();
  const add = (a: number, b: number, c: number, d: number): void => {
    // Canonicalised by the outer pair, exactly as OpenMM's branches do: a
    // torsion and its reverse are the same torsion.
    unique.add(a < d ? `${a},${b},${c},${d}` : `${d},${c},${b},${a}`);
  };
  for (const angle of angles) {
    const [first, middle, last] = angle;
    for (const atom of neighbours[first] as readonly number[]) {
      if (atom !== middle && atom !== last) add(atom, first, middle, last);
    }
    for (const atom of neighbours[last] as readonly number[]) {
      if (atom !== first && atom !== middle) add(first, middle, last, atom);
    }
  }
  return sortTuples(unique) as readonly Quad[];
}

/**
 * Every (central, three of its neighbours) for an atom with three or more
 * bonds. Not deduplicated and not sorted: OpenMM appends these in atom order
 * and takes `itertools.combinations` of the neighbour list, and the order
 * reaches the output because an improper's atom order is derived from it.
 */
export function findImpropers(
  neighbours: readonly (readonly number[])[],
): readonly Quad[] {
  const impropers: Quad[] = [];
  for (const [atom, bonded] of neighbours.entries()) {
    if (bonded.length <= 2) continue;
    for (let i = 0; i < bonded.length; i++) {
      for (let j = i + 1; j < bonded.length; j++) {
        for (let k = j + 1; k < bonded.length; k++) {
          impropers.push([
            atom,
            bonded[i] as number,
            bonded[j] as number,
            bonded[k] as number,
          ]);
        }
      }
    }
  }
  return impropers;
}

function sortTuples(unique: ReadonlySet<string>): readonly (readonly number[])[] {
  const tuples = [...unique].map((key) => key.split(",").map(Number));
  // Lexicographic, which is what Python's `sorted` on tuples of ints gives.
  tuples.sort((a, b) => {
    for (let i = 0; i < a.length; i++) {
      const difference = (a[i] as number) - (b[i] as number);
      if (difference !== 0) return difference;
    }
    return 0;
  });
  return tuples;
}

/**
 * Expand one row's keys into type sets.
 *
 * `null` in `keys` is an empty class, which `_findAtomTypes` resolves to
 * `_atomClasses['']` -- the set of every type -- and which the matchers then
 * detect by identity to mean "wildcard". We carry the wildcard as `null`
 * instead, which makes the specificity test a null check rather than a
 * set comparison.
 */
export function resolveSlots(row: ParameterRow, field: ForceField): readonly Slot[] {
  return row.keys.map((key, index) => {
    if (key === null) return null;
    if (row.byClass[index] === true) {
      return new Set(field.typesByClass.get(key) ?? []);
    }
    return new Set([key]);
  });
}

function accepts(slot: Slot, type: string): boolean {
  return slot === null || slot.has(type);
}

function hasWildcard(slots: readonly Slot[]): boolean {
  return slots.some((slot) => slot === null);
}

interface Resolved<R> {
  readonly row: R;
  readonly slots: readonly Slot[];
  readonly wildcard: boolean;
}

function resolveAll<R extends ParameterRow>(
  rows: readonly R[],
  field: ForceField,
): readonly Resolved<R>[] {
  return rows.map((row) => {
    const slots = resolveSlots(row, field);
    return { row, slots, wildcard: hasWildcard(slots) };
  });
}

/**
 * Find the row that applies, preferring a non-wildcard match.
 *
 * `matches` says whether a row's slots fit the term; the caller supplies it
 * because the symmetry differs per term type. Returns the row and how many
 * others matched at the same specificity, which is the ambiguity the module
 * comment describes.
 */
function bestMatch<R extends ParameterRow>(
  resolved: readonly Resolved<R>[],
  matches: (slots: readonly Slot[]) => boolean,
): { row: R | null; ties: number } {
  let specific: R | null = null;
  let wildcard: R | null = null;
  let specificTies = 0;
  let wildcardTies = 0;
  for (const candidate of resolved) {
    if (!matches(candidate.slots)) continue;
    if (candidate.wildcard) {
      if (wildcard === null) wildcard = candidate.row;
      else if (!sameValues(wildcard, candidate.row)) wildcardTies++;
    } else {
      if (specific === null) specific = candidate.row;
      else if (!sameValues(specific, candidate.row)) specificTies++;
    }
  }
  if (specific !== null) return { row: specific, ties: specificTies };
  return { row: wildcard, ties: wildcardTies };
}

/** Whether two rows would give the same term, so a tie between them is harmless. */
function sameValues(a: ParameterRow, b: ParameterRow): boolean {
  const first = a as TorsionRow;
  const second = b as TorsionRow;
  if (first.terms !== undefined && second.terms !== undefined) {
    if (first.terms.length !== second.terms.length) return false;
    return first.terms.every((term, index) => {
      const other = second.terms[index] as { k: number; periodicity: number; phase: number };
      return (
        term.k === other.k &&
        term.periodicity === other.periodicity &&
        term.phase === other.phase
      );
    });
  }
  const keys = new Set([...Object.keys(a.values), ...Object.keys(b.values)]);
  for (const key of keys) {
    if (a.values[key] !== b.values[key]) return false;
  }
  return true;
}

export function assignBonds(
  bonds: readonly Pair[],
  types: readonly string[],
  field: ForceField,
  diagnostics: MatchDiagnostics = { ambiguous: 0, unmatched: 0 },
): { terms: readonly HarmonicTerm[]; diagnostics: MatchDiagnostics } {
  const resolved = resolveAll(field.bonds, field);
  const terms: HarmonicTerm[] = [];
  for (const [a, b] of bonds) {
    const type1 = types[a] as string;
    const type2 = types[b] as string;
    const { row, ties } = bestMatch(
      resolved,
      ([first, second]) =>
        (accepts(first ?? null, type1) && accepts(second ?? null, type2)) ||
        (accepts(first ?? null, type2) && accepts(second ?? null, type1)),
    );
    diagnostics.ambiguous += ties;
    if (row === null) {
      diagnostics.unmatched++;
      continue;
    }
    const k = row.values["k"] ?? 0;
    // k == 0 means the force field deliberately leaves the term out, which is
    // not the same as it being absent: `createForce` skips those.
    if (k === 0) continue;
    terms.push({ atoms: [a, b], value: row.values["length"] ?? 0, k });
  }
  return { terms, diagnostics };
}

export function assignAngles(
  angles: readonly Triple[],
  types: readonly string[],
  field: ForceField,
  diagnostics: MatchDiagnostics = { ambiguous: 0, unmatched: 0 },
): { terms: readonly HarmonicTerm[]; diagnostics: MatchDiagnostics } {
  const resolved = resolveAll(field.angles, field);
  const terms: HarmonicTerm[] = [];
  for (const [a, b, c] of angles) {
    const type1 = types[a] as string;
    const type2 = types[b] as string;
    const type3 = types[c] as string;
    const { row, ties } = bestMatch(resolved, ([first, second, third]) => {
      // The middle slot is fixed; only the ends swap.
      if (!accepts(second ?? null, type2)) return false;
      return (
        (accepts(first ?? null, type1) && accepts(third ?? null, type3)) ||
        (accepts(first ?? null, type3) && accepts(third ?? null, type1))
      );
    });
    diagnostics.ambiguous += ties;
    if (row === null) {
      diagnostics.unmatched++;
      continue;
    }
    const k = row.values["k"] ?? 0;
    if (k === 0) continue;
    terms.push({ atoms: [a, b, c], value: row.values["angle"] ?? 0, k });
  }
  return { terms, diagnostics };
}

export function assignPropers(
  propers: readonly Quad[],
  types: readonly string[],
  field: ForceField,
  diagnostics: MatchDiagnostics = { ambiguous: 0, unmatched: 0 },
): { terms: readonly TorsionTerm[]; diagnostics: MatchDiagnostics } {
  const resolved = resolveAll(field.propers, field);
  const terms: TorsionTerm[] = [];
  for (const quad of propers) {
    const [type1, type2, type3, type4] = quad.map((atom) => types[atom] as string) as [
      string,
      string,
      string,
      string,
    ];
    const { row, ties } = bestMatch(resolved, ([first, second, third, fourth]) => {
      const forwards =
        accepts(first ?? null, type1) &&
        accepts(second ?? null, type2) &&
        accepts(third ?? null, type3) &&
        accepts(fourth ?? null, type4);
      const backwards =
        accepts(first ?? null, type4) &&
        accepts(second ?? null, type3) &&
        accepts(third ?? null, type2) &&
        accepts(fourth ?? null, type1);
      return forwards || backwards;
    });
    diagnostics.ambiguous += ties;
    if (row === null) {
      diagnostics.unmatched++;
      continue;
    }
    for (const term of row.terms) {
      if (term.k === 0) continue;
      terms.push({
        atoms: quad,
        periodicity: term.periodicity,
        phase: term.phase,
        k: term.k,
      });
    }
  }
  return { terms, diagnostics };
}
