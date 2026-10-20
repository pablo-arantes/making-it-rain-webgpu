/**
 * The whole parameter assignment, in one call.
 *
 * Everything in this directory is one pipeline: read a PDB, reconstruct its
 * bonds, match each residue to a template, read off types and charges, walk the
 * graph for bonds/angles/torsions, and build the nonbonded exceptions. The
 * pieces are separate files because they are tested separately -- a bond
 * reconstruction bug and a template matching bug look nothing alike and should
 * not be debugged together -- but callers want the whole thing.
 *
 * The order is forced: template matching needs the bonds, the improper ordering
 * rules need the template correspondence, and the exceptions need the charges.
 * Nothing here is optional or reorderable.
 */

import type { PdbTopology } from "../analysis/pdb.js";
import { assignAtoms, atomContexts, type Assignment } from "./assign.js";
import {
  assignAngles,
  assignBonds,
  assignPropers,
  findAngles,
  findImpropers,
  findPropers,
  neighbourLists,
  type HarmonicTerm,
  type MatchDiagnostics,
  type Pair,
  type TorsionTerm,
} from "./bonded.js";
import { inferBonds, type StandardBondTable } from "./bonds.js";
import { findExceptions, type Exception } from "./exceptions.js";
import type { ForceField } from "./ffxml.js";
import { assignImpropers } from "./torsions.js";

export interface PreparedSystem {
  readonly topology: PdbTopology;
  readonly bonds: readonly Pair[];
  readonly assignment: Assignment;
  readonly bondTerms: readonly HarmonicTerm[];
  readonly angleTerms: readonly HarmonicTerm[];
  /** Propers and impropers together, as OpenMM's single force holds them. */
  readonly torsionTerms: readonly TorsionTerm[];
  readonly exceptions: readonly Exception[];
  readonly diagnostics: {
    readonly bonds: MatchDiagnostics;
    readonly angles: MatchDiagnostics;
    readonly propers: MatchDiagnostics;
  };
}

export function prepareSystem(
  topology: PdbTopology,
  field: ForceField,
  standardBonds: StandardBondTable,
): PreparedSystem {
  const bonds = inferBonds(topology, standardBonds);
  const assignment = assignAtoms(topology, bonds, field);
  const types = assignment.atoms.map((atom) => atom.type);
  const context = atomContexts(topology, assignment);

  const neighbours = neighbourLists(topology.atoms.length, bonds);
  const angles = findAngles(bonds, neighbours);
  const propers = findPropers(angles, neighbours);
  const impropers = findImpropers(neighbours);

  const bondResult = assignBonds(bonds, types, field);
  const angleResult = assignAngles(angles, types, field);
  const properResult = assignPropers(propers, types, field);

  return {
    topology,
    bonds,
    assignment,
    bondTerms: bondResult.terms,
    angleTerms: angleResult.terms,
    // Propers first, then impropers: the order a single `PeriodicTorsionForce`
    // receives them, since `createForce` adds all propers before any improper.
    torsionTerms: [
      ...properResult.terms,
      ...assignImpropers(impropers, context, field),
    ],
    exceptions: findExceptions(
      assignment.atoms,
      bonds,
      field.coulomb14Scale,
      field.lj14Scale,
    ),
    diagnostics: {
      bonds: bondResult.diagnostics,
      angles: angleResult.diagnostics,
      propers: properResult.diagnostics,
    },
  };
}
