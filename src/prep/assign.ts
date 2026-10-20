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
 * - **charge** comes from whichever of the two the force field says. ff14SB's
 *   `<NonbondedForce>` carries `<UseAttributeFromResidue name="charge"/>`, so
 *   the charge is on the *residue template*; ff99SB and ff99SB-ILDN put it on
 *   the atom type instead. Both sources must be kept, because Amber gives the
 *   same type different charges in different residues -- a backbone `N` is
 *   -0.4157 in alanine and -0.5163 in the N-terminal variant -- so reading
 *   ff14SB's charges from the type table would flatten them.
 *
 * 🔴 The first version of this file read the flag and then fell back to **zero**
 * when it was absent, with a comment claiming the other case was handled. It
 * was not: under ff99SB-ILDN every charge came out zero while the masses,
 * bonds, angles, torsions and template matches all still agreed with OpenMM.
 * Nothing in a one-force-field test could see it. That is the whole argument
 * for running the gate against a second file.
 */

import type { PdbTopology } from "../analysis/pdb.js";
import type { AtomContext } from "./torsions.js";
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
 * The charge a type carries, or a refusal.
 *
 * Defaulting to zero here is what hid the ff99SB-ILDN bug, so a force field
 * that provides charge through neither route is an error rather than a
 * neutral molecule. A silent zero charge is the worst case: the system runs,
 * the energies look plausible, and the electrostatics are simply absent.
 */
function requireCharge(charge: number | null, type: string, atom: number): number {
  if (charge === null) {
    throw new MissingParameters(
      "a charge -- the force field neither puts one on the atom type nor " +
        'declares <UseAttributeFromResidue name="charge"/>',
      type,
      atom,
    );
  }
  return charge;
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
        charge: chargeFromResidue
          ? (match.charges[position] as number)
          : requireCharge(lj.charge, type, atom),
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

/**
 * Per-atom context for the improper ordering rules.
 *
 * Built here because the template index is only available while the match is
 * in hand: it is the atom's position *inside the template it matched*, which
 * no later pass can recover. `ordering="amber"` breaks ties on it, so an
 * improper's atom order -- and therefore the sign of its dihedral angle --
 * depends on getting it from the match rather than from the file.
 */
export function atomContexts(
  topology: PdbTopology,
  assignment: Assignment,
): readonly AtomContext[] {
  const contexts: (AtomContext | undefined)[] = assignment.atoms.map(() => undefined);
  for (const match of assignment.matches) {
    const residue = topology.residues[match.residue] as { atoms: readonly number[] };
    for (const [position, atom] of residue.atoms.entries()) {
      const assigned = assignment.atoms[atom] as AssignedAtom;
      contexts[atom] = {
        type: assigned.type,
        element: (topology.atoms[atom] as { element: string }).element,
        mass: assigned.mass,
        residue: match.residue,
        templateIndex: match.atomMatches[position] as number,
      };
    }
  }
  return contexts.map((context, index) => {
    if (context === undefined) throw new Error(`atom ${index} belongs to no residue`);
    return context;
  });
}

/** Total charge, which should be an integer to within round-off. */
export function totalCharge(assignment: Assignment): number {
  let total = 0;
  for (const atom of assignment.atoms) total += atom.charge;
  return total;
}
