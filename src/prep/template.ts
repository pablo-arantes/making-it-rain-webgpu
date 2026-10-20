/**
 * Matching residues to force field templates.
 *
 * This is where atom types and charges come from, and it is **graph matching**,
 * not name matching. A PDB says `ALA` for every alanine; ff14SB has `ALA`,
 * `NALA` and `CALA`, and which one applies is decided by the shape of the
 * residue -- three hydrogens on the nitrogen means N-terminal, an extra oxygen
 * on the carbonyl means C-terminal. Matching by name would assign the wrong
 * charges to both ends of every chain, and the energy would be wrong by
 * hundreds of kJ/mol with nothing to point at.
 *
 * Transcribed from `ForceField._getResidueTemplateMatches` and
 * `_findAtomMatches` in OpenMM's `forcefield.py`. Four things it does that are
 * not obvious:
 *
 * **Templates are indexed by an element signature.** The sorted multiset of
 * element symbols. Only templates with the same signature are tried, which
 * turns "search 78 templates" into "search two or three" and is also what makes
 * the atom-count check redundant.
 *
 * **Atoms match by element, except where the template gives none.** A template
 * atom with no element is a virtual site, and those match by *name* instead.
 *
 * **The external bond count has to agree.** An atom bonded to the next residue
 * has one external bond, and that is the whole difference between `ALA` and
 * `CALA` once the hydrogens are accounted for. Ignoring it makes a mid-chain
 * residue match a terminal template.
 *
 * **Several templates may match, and that is allowed** -- but only if they
 * assign identical types and charges to every atom. Otherwise the residue is
 * ambiguous and refusing is the only safe answer, because picking one silently
 * chooses a protonation state on the user's behalf.
 */

import type { PdbTopology } from "../analysis/pdb.js";
import type { ForceField, ResidueTemplate } from "./ffxml.js";

/** A template with its graph precomputed, which the matcher needs per atom. */
export interface PreparedTemplate {
  readonly template: ResidueTemplate;
  readonly signature: string;
  readonly elements: readonly (string | null)[];
  /** Indices of the template atoms each atom is bonded to. */
  readonly bondedTo: readonly (readonly number[])[];
  readonly externalBonds: readonly number[];
}

export interface ResidueMatch {
  readonly residue: number;
  readonly template: string;
  /** For each atom of the residue, the index of the template atom it matched. */
  readonly atomMatches: readonly number[];
  /** Per atom of the residue, in residue order. */
  readonly types: readonly string[];
  readonly charges: readonly number[];
}

export class TemplateMatchFailed extends Error {
  constructor(
    readonly residueName: string,
    readonly residueId: string,
    readonly detail: string,
  ) {
    super(
      `no force field template matches residue ${residueName} ${residueId}: ` +
        `${detail}.\n\n` +
        "Templates are matched by shape rather than by name, so this means the " +
        "residue's atoms and bonds do not correspond to anything the force " +
        "field describes -- most often a missing atom, an extra hydrogen, a " +
        "non-standard residue, or a protonation state the force field does not " +
        "carry. Preparing the structure first (adding missing atoms and " +
        "hydrogens) is what resolves it; guessing a template would assign " +
        "silently wrong charges.",
    );
    this.name = "TemplateMatchFailed";
  }
}

export class AmbiguousTemplate extends Error {
  constructor(residueName: string, residueId: string, names: readonly string[]) {
    super(
      `residue ${residueName} ${residueId} matches ${names.length} templates ` +
        `that assign different parameters (${names.join(", ")}). ` +
        "Choosing one would pick a protonation state on your behalf; rename " +
        "the residue to the variant you mean.",
    );
    this.name = "AmbiguousTemplate";
  }
}

/**
 * The sorted element multiset, as OpenMM's `_createResidueSignature` makes it.
 *
 * Exported because it is also the answer to "why did this residue not match":
 * a residue whose signature appears in no template cannot match anything, and
 * saying so is more useful than listing 78 templates that failed.
 */
export function residueSignature(elements: readonly (string | null)[]): string {
  const counts = new Map<string, number>();
  for (const element of elements) {
    const key = element ?? "";
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([element, count]) => `${element}${count}`)
    .join("");
}

function prepareTemplate(
  template: ResidueTemplate,
  field: ForceField,
): PreparedTemplate {
  const elements = template.atoms.map(
    (atom) => field.types.get(atom.type)?.element ?? null,
  );
  const bondedTo: number[][] = template.atoms.map(() => []);
  for (const [a, b] of template.bonds) {
    if (a < 0 || b < 0) continue;
    (bondedTo[a] as number[]).push(b);
    (bondedTo[b] as number[]).push(a);
  }
  const externalBonds = template.atoms.map(() => 0);
  for (const atom of template.externalBonds) {
    if (atom >= 0) externalBonds[atom] = (externalBonds[atom] as number) + 1;
  }
  return {
    template,
    signature: residueSignature(elements),
    elements,
    bondedTo,
    externalBonds,
  };
}

/** Index the templates by signature, once per force field. */
export function prepareTemplates(
  field: ForceField,
): ReadonlyMap<string, readonly PreparedTemplate[]> {
  const bySignature = new Map<string, PreparedTemplate[]>();
  for (const template of field.templates) {
    const prepared = prepareTemplate(template, field);
    const existing = bySignature.get(prepared.signature);
    if (existing === undefined) bySignature.set(prepared.signature, [prepared]);
    else existing.push(prepared);
  }
  return bySignature;
}

/**
 * Backtracking search for an atom-by-atom correspondence.
 *
 * Transcribed from `_findAtomMatches`. The bond-consistency test only looks at
 * neighbours already placed (`bonded < position`), which is what keeps this
 * from being exponential on a real residue: a wrong partial assignment is
 * rejected as soon as it contradicts a bond rather than at the end.
 */
function findAtomMatches(
  elements: readonly string[],
  names: readonly string[],
  prepared: PreparedTemplate,
  bondedTo: readonly (readonly number[])[],
  externalBonds: readonly number[],
  matches: (number | null)[],
  taken: boolean[],
  position: number,
): boolean {
  if (position === elements.length) return true;
  const element = elements[position] as string;
  const name = names[position] as string;
  for (let candidate = 0; candidate < prepared.template.atoms.length; candidate++) {
    if (taken[candidate] === true) continue;
    const templateElement = prepared.elements[candidate] ?? null;
    const matchesAtom =
      templateElement !== null
        ? templateElement.toUpperCase() === element.toUpperCase()
        : // No element on the template atom means a virtual site, matched by name.
          (prepared.template.atoms[candidate] as { name: string }).name === name;
    if (!matchesAtom) continue;
    if (
      (prepared.bondedTo[candidate] as readonly number[]).length !==
      (bondedTo[position] as readonly number[]).length
    ) {
      continue;
    }
    if (prepared.externalBonds[candidate] !== externalBonds[position]) continue;

    const consistent = (bondedTo[position] as readonly number[]).every(
      (neighbour) =>
        neighbour > position ||
        (prepared.bondedTo[candidate] as readonly number[]).includes(
          matches[neighbour] as number,
        ),
    );
    if (!consistent) continue;

    matches[position] = candidate;
    taken[candidate] = true;
    if (
      findAtomMatches(
        elements,
        names,
        prepared,
        bondedTo,
        externalBonds,
        matches,
        taken,
        position + 1,
      )
    ) {
      return true;
    }
    matches[position] = null;
    taken[candidate] = false;
  }
  return false;
}

/** Whether two matches assign the same type and charge to every atom. */
function parametersIdentical(
  first: { prepared: PreparedTemplate; matches: readonly number[] },
  second: { prepared: PreparedTemplate; matches: readonly number[] },
): boolean {
  for (let atom = 0; atom < first.matches.length; atom++) {
    const a = first.prepared.template.atoms[first.matches[atom] as number];
    const b = second.prepared.template.atoms[second.matches[atom] as number];
    if (a === undefined || b === undefined) return false;
    if (a.type !== b.type || a.charge !== b.charge) return false;
  }
  return true;
}

/**
 * Match every residue, returning the types and charges its atoms take.
 *
 * `bonds` is the topology's bond list; the matcher needs it to count each
 * atom's internal and external bonds, which is what distinguishes a terminal
 * residue from a mid-chain one.
 */
export function matchResidues(
  topology: PdbTopology,
  bonds: readonly (readonly [number, number])[],
  field: ForceField,
  bySignature: ReadonlyMap<string, readonly PreparedTemplate[]> = prepareTemplates(field),
): readonly ResidueMatch[] {
  const neighbours: number[][] = topology.atoms.map(() => []);
  for (const [a, b] of bonds) {
    (neighbours[a] as number[]).push(b);
    (neighbours[b] as number[]).push(a);
  }

  const results: ResidueMatch[] = [];
  for (const [residueIndex, residue] of topology.residues.entries()) {
    const local = new Map<number, number>();
    for (const [position, atom] of residue.atoms.entries()) local.set(atom, position);

    const elements = residue.atoms.map(
      (atom) => (topology.atoms[atom] as { element: string }).element,
    );
    const names = residue.atoms.map(
      (atom) => (topology.atoms[atom] as { name: string }).name,
    );
    const bondedTo: number[][] = [];
    const externalBonds: number[] = [];
    for (const atom of residue.atoms) {
      const inside: number[] = [];
      let outside = 0;
      for (const other of neighbours[atom] as number[]) {
        const position = local.get(other);
        if (position === undefined) outside++;
        else inside.push(position);
      }
      bondedTo.push(inside);
      externalBonds.push(outside);
    }

    const signature = residueSignature(elements);
    const candidates = bySignature.get(signature) ?? [];
    const found: { prepared: PreparedTemplate; matches: number[] }[] = [];
    for (const prepared of candidates) {
      const matches: (number | null)[] = residue.atoms.map(() => null);
      const taken = prepared.template.atoms.map(() => false);
      if (
        findAtomMatches(
          elements,
          names,
          prepared,
          bondedTo,
          externalBonds,
          matches,
          taken,
          0,
        )
      ) {
        found.push({ prepared, matches: matches as number[] });
      }
    }

    if (found.length === 0) {
      // Say what was looked for, since "no match" alone sends the reader to the
      // wrong place: the signature and the external bond count are what
      // actually failed.
      const detail =
        candidates.length === 0
          ? `no template has its element signature (${signature})`
          : `${candidates.length} template(s) share its element signature ` +
            `(${candidates.map((c) => c.template.name).join(", ")}) but none ` +
            `matches its bonds — it has ${externalBonds.reduce((a, b) => a + b, 0)} ` +
            "bond(s) to neighbouring residues";
      throw new TemplateMatchFailed(residue.name, residue.id, detail);
    }
    if (found.length > 1) {
      const first = found[0] as { prepared: PreparedTemplate; matches: number[] };
      for (const other of found.slice(1)) {
        if (!parametersIdentical(first, other)) {
          throw new AmbiguousTemplate(
            residue.name,
            residue.id,
            found.map((f) => f.prepared.template.name),
          );
        }
      }
    }

    const chosen = found[0] as { prepared: PreparedTemplate; matches: number[] };
    results.push({
      residue: residueIndex,
      template: chosen.prepared.template.name,
      atomMatches: chosen.matches,
      types: chosen.matches.map(
        (index) => (chosen.prepared.template.atoms[index] as { type: string }).type,
      ),
      charges: chosen.matches.map(
        (index) =>
          (chosen.prepared.template.atoms[index] as { charge: number | null }).charge ?? 0,
      ),
    });
  }
  return results;
}
