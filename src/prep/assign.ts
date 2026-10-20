/**
 * Per-atom parameters: mass, charge, sigma, epsilon.
 *
 * Once a residue is matched to a template, every atom has a type name, and that
 * name is the key to everything else. Three sources, and they are deliberately
 * not the same source:
 *
 * - **mass** comes from the `<Type>` entry, the force field's own table.
 * - **sigma and epsilon** come from `<NonbondedForce><Atom>`, keyed by type or
 *   by class; `parseForceField` has already expanded class-keyed rows onto
 *   their types, with a type-keyed row winning.
 * - **charge** comes from the *residue template*, not from the nonbonded table,
 *   because ff14SB's `<NonbondedForce>` carries
 *   `<UseAttributeFromResidue name="charge"/>`. This is not a detail: Amber
 *   gives the same atom type different charges in different residues -- a
 *   backbone `N` is -0.4157 in alanine and -0.5163 in the N-terminal variant --
 *   so reading charge from the type table would quietly flatten them and leave
 *   the total charge wrong.
 *
 * `UseAttributeFromResidue` is checked rather than assumed, so a force field
 * that puts charge on the type is still handled correctly instead of silently
 * getting zeros.
 */

import type { PdbTopology } from "../analysis/pdb.js";
import type { ForceField } from "./ffxml.js";
import { matchResidues, prepareTemplates, type ResidueMatch } from "./template.js";

export interface AssignedAtom {
  readonly type: string;
  readonly className: string;
  /** amu. */
  readonly mass: number;
  /** e. */
  readonly charge: number;
  /** nm. */
  readonly sigma: number;
  /** kJ/mol. */
  readonly epsilon: number;
}

export interface Assignment {
  readonly atoms: readonly AssignedAtom[];
  readonly matches: readonly ResidueMatch[];
  /** Per atom, the class name, which is what bonded tables are usually keyed by. */
  readonly classes: readonly string[];
}

export class MissingParameters extends Error {
  constructor(what: string, type: string, atom: number) {
    super(
      `the force field has no ${what} for atom type ${type} (atom ${atom}). ` +
        "The residue matched a template, so this is a gap in the force field " +
        "files rather than a problem with the structure: a template referring " +
        "to a type that no table covers. Energies would be wrong by whatever " +
        "that atom contributes, so it is refused instead.",
    );
    this.name = "MissingParameters";
  }
}

/**
 * Match every residue and assign each atom its own parameters.
 *
 * Returns the matches alongside the atoms, because bonded assignment needs the
 * types and the template correspondence, and running the match twice would be
 * both slower and a chance for the two runs to disagree.
 */
export function assignAtoms(
  topology: PdbTopology,
  bonds: readonly (readonly [number, number])[],
  field: ForceField,
): Assignment {
  const matches = matchResidues(topology, bonds, field, prepareTemplates(field));
  const chargeFromResidue = field.fromResidue.includes("charge");

  const atoms: (AssignedAtom | undefined)[] = topology.atoms.map(() => undefined);
  for (const match of matches) {
    const residue = topology.residues[match.residue] as { atoms: readonly number[] };
    for (const [position, atom] of residue.atoms.entries()) {
      const type = match.types[position] as string;
      const entry = field.types.get(type);
      if (entry === undefined) throw new MissingParameters("mass or element", type, atom);
      const lj = field.nonbonded.get(type);
      if (lj === undefined) {
        throw new MissingParameters("Lennard-Jones parameters", type, atom);
      }
      atoms[atom] = {
        type,
        className: entry.className,
        mass: entry.mass,
        charge: chargeFromResidue ? (match.charges[position] as number) : 0,
        sigma: lj.sigma,
        epsilon: lj.epsilon,
      };
    }
  }

  const assigned = atoms.map((atom, index) => {
    if (atom === undefined) {
      // Cannot happen while every atom belongs to a residue, which `readPdb`
      // guarantees -- but a silent `undefined` here would become a NaN charge
      // thousands of lines later, so it is caught where it is cheap to explain.
      throw new Error(`atom ${index} belongs to no residue`);
    }
    return atom;
  });

  return {
    atoms: assigned,
    matches,
    classes: assigned.map((atom) => atom.className),
  };
}

/** Total charge, which should be an integer to within round-off. */
export function totalCharge(assignment: Assignment): number {
  let total = 0;
  for (const atom of assignment.atoms) total += atom.charge;
  return total;
}
