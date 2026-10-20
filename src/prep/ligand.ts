/**
 * Finding a ligand in a PDB, and giving it the chemistry a force field needs.
 *
 * Phase 4's front end. Three jobs, in order:
 *
 * 1. **Which residues could be a ligand.** A PDB's `HETATM` records hold the
 *    ligand, the water, the ions, the cryoprotectant and whatever else was in
 *    the drop. Only some of that is wanted, and which is a decision for the
 *    user, so this reports candidates with enough detail to choose rather than
 *    guessing.
 * 2. **Bond orders, from the CCD.** See `src/prep/ccd.ts`: a PDB has none, and
 *    the component definition is the only authority.
 * 3. **What is missing.** A deposited ligand is routinely incomplete — no
 *    hydrogens, sometimes a disordered tail — and the difference between
 *    "absent because it is a leaving atom" and "absent because the density was
 *    poor" decides whether the molecule can be typed at all.
 */

import type { PdbAtom, PdbResidue, PdbTopology } from "../analysis/pdb.js";
import type { BondOrder, ChemicalComponent } from "./ccd.js";
import type { Pair } from "./bonded.js";

/**
 * Residue names that are never the ligand of interest.
 *
 * Water, the common counter-ions, and the usual crystallisation additives.
 * Deliberately a *ranking* input rather than a filter: a structure can have
 * glycerol as its actual ligand, so these are reported as unlikely rather than
 * hidden.
 */
const COMMON_ADDITIVES = new Set([
  "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "FE2",
  "CU", "NI", "CO", "CD", "HG", "BR", "IOD", "F", "LI", "RB", "CS", "SR", "BA",
  "SO4", "PO4", "NO3", "ACT", "EDO", "GOL", "PEG", "PGE", "MPD", "DMS", "TRS",
  "EPE", "MES", "IMD", "FMT", "ACY", "CIT", "TLA", "BME", "DTT", "AZI", "CO3",
]);

/** The twenty standard residues plus the usual variants and caps. */
const POLYMER = new Set([
  "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
  "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE",
  "HIP", "CYX", "ASH", "GLH", "LYN", "ACE", "NME", "NHE", "MSE",
  "A", "C", "G", "U", "DA", "DC", "DG", "DT", "DI", "N",
]);

export interface LigandCandidate {
  /** Index into `topology.residues`. */
  readonly residue: number;
  /** The three-letter component code, which is the CCD key. */
  readonly code: string;
  readonly chainId: string;
  readonly residueId: string;
  readonly heavyAtoms: number;
  readonly hydrogens: number;
  readonly elements: readonly string[];
  /**
   * Whether this looks like a crystallisation additive or an ion rather than
   * the ligand. Advisory: a structure can have glycerol as its real ligand, so
   * nothing is hidden on the strength of it.
   */
  readonly likelyAdditive: boolean;
}

/**
 * Every residue that is not polymer and not obviously solvent, largest first.
 *
 * Ordered by heavy-atom count because the interesting ligand is almost always
 * the biggest non-polymer thing present, which makes the default choice right
 * most of the time without hiding the alternatives.
 */
export function findLigandCandidates(
  topology: PdbTopology,
): readonly LigandCandidate[] {
  const found: LigandCandidate[] = [];
  for (const [index, residue] of topology.residues.entries()) {
    if (POLYMER.has(residue.name)) continue;
    const atoms = residue.atoms.map((a) => topology.atoms[a] as PdbAtom);
    const heavy = atoms.filter((a) => a.element.toUpperCase() !== "H");
    // A single atom is an ion, and worth listing -- someone may want to keep a
    // catalytic zinc -- but it is never typed by the ligand path.
    found.push({
      residue: index,
      code: residue.name,
      chainId: residue.chainId,
      residueId: residue.id,
      heavyAtoms: heavy.length,
      hydrogens: atoms.length - heavy.length,
      elements: [...new Set(heavy.map((a) => a.element.toUpperCase()))].sort(),
      likelyAdditive: COMMON_ADDITIVES.has(residue.name) || heavy.length === 1,
    });
  }
  return found.sort(
    (a, b) =>
      Number(a.likelyAdditive) - Number(b.likelyAdditive) ||
      b.heavyAtoms - a.heavyAtoms,
  );
}

export interface LigandBond {
  readonly atoms: Pair;
  readonly order: BondOrder;
  readonly aromatic: boolean;
}

export interface MatchedLigand {
  readonly code: string;
  /** Topology atom indices, in the order the file has them. */
  readonly atoms: readonly number[];
  readonly bonds: readonly LigandBond[];
  /** Per atom of the ligand, from the component definition. */
  readonly formalCharges: readonly number[];
  readonly aromatic: readonly boolean[];
  readonly totalCharge: number;
  /**
   * Component atoms the fragment does not have.
   *
   * 🔴 The element is carried because the three cases are not alike, and an
   * undifferentiated count is useless. A deposited ligand almost never has
   * hydrogens, so those are expected; a *leaving* atom is absent by definition;
   * a missing **heavy** atom means the model is incomplete and the molecule
   * cannot be typed as the component it claims to be. JZ4 in 3HTB reports 12
   * missing atoms and all twelve are hydrogens.
   */
  readonly missing: readonly {
    readonly name: string;
    readonly element: string;
    readonly leaving: boolean;
  }[];
  /** The count that actually indicates a problem. */
  readonly missingHeavyAtoms: number;
  /** Fragment atoms the component does not define, which is always a problem. */
  readonly unexpected: readonly string[];
}

export class LigandMatchFailed extends Error {
  constructor(code: string, detail: string) {
    super(
      `the ligand ${code} cannot be matched to its CCD definition: ${detail}.\n\n` +
        "Bond orders come from the component definition and are transferred by " +
        "atom name, so the names in the file have to be the CCD's own. A " +
        "ligand renamed by a modelling program, or one extracted without its " +
        "names, needs an SDF or a MOL2 supplied directly instead.",
    );
    this.name = "LigandMatchFailed";
  }
}

/**
 * Transfer the component's chemistry onto one residue of the topology.
 *
 * 🔴 By **atom name**, not by graph. The names in a deposited `HETATM` record
 * are the CCD's `atom_id` values, so the correspondence is already in the file
 * and does not need to be inferred. That matters beyond simplicity: a
 * substructure match has to choose among symmetry-equivalent mappings, and for
 * a symmetric ligand the choice is arbitrary — it gives the right bond orders
 * but can attach them to the wrong atoms, which then disagrees with anything
 * else keyed by name.
 *
 * Hydrogens in the fragment are kept and bonded, but the component's hydrogens
 * are not added: a deposited ligand usually has none, and placing them is a
 * separate job with its own geometry.
 */
export function matchLigand(
  topology: PdbTopology,
  residueIndex: number,
  component: ChemicalComponent,
): MatchedLigand {
  const residue = topology.residues[residueIndex] as PdbResidue;
  if (residue.name.toUpperCase() !== component.id.toUpperCase()) {
    throw new LigandMatchFailed(
      component.id,
      `the residue is called ${residue.name} and the component is ${component.id}`,
    );
  }

  const byName = new Map<string, number>();
  const duplicates: string[] = [];
  for (const atom of residue.atoms) {
    const name = (topology.atoms[atom] as PdbAtom).name;
    if (byName.has(name)) duplicates.push(name);
    else byName.set(name, atom);
  }
  if (duplicates.length > 0) {
    // Two atoms with one name make the name-keyed transfer ambiguous, and
    // picking either is a coin toss that nothing downstream could detect.
    throw new LigandMatchFailed(
      component.id,
      `atom name${duplicates.length === 1 ? "" : "s"} ` +
        `${[...new Set(duplicates)].join(", ")} appear${
          duplicates.length === 1 ? "s" : ""
        } more than once, so a name cannot identify an atom`,
    );
  }

  const defined = new Map(component.atoms.map((atom) => [atom.name, atom]));
  const unexpected: string[] = [];
  for (const [name] of byName) {
    if (!defined.has(name)) unexpected.push(name);
  }
  if (unexpected.length > 0) {
    throw new LigandMatchFailed(
      component.id,
      `the file has atom${unexpected.length === 1 ? "" : "s"} ` +
        `${unexpected.join(", ")} that ${component.id} does not define`,
    );
  }

  const missing = component.atoms
    .filter((atom) => !byName.has(atom.name))
    .map((atom) => ({
      name: atom.name,
      element: atom.element,
      leaving: atom.leaving,
    }));
  const missingHeavyAtoms = missing.filter(
    (atom) => atom.element.toUpperCase() !== "H" && !atom.leaving,
  ).length;

  const bonds: LigandBond[] = [];
  for (const bond of component.bonds) {
    const a = byName.get(bond.from);
    const b = byName.get(bond.to);
    // A bond to an atom the fragment lacks is simply absent, which is right
    // for a leaving atom and is reported through `missing` otherwise.
    if (a === undefined || b === undefined) continue;
    bonds.push({ atoms: [a, b], order: bond.order, aromatic: bond.aromatic });
  }

  const atoms = [...residue.atoms];
  const formalCharges = atoms.map(
    (atom) => defined.get((topology.atoms[atom] as PdbAtom).name)?.formalCharge ?? 0,
  );
  const aromatic = atoms.map(
    (atom) => defined.get((topology.atoms[atom] as PdbAtom).name)?.aromatic ?? false,
  );
  return {
    code: component.id,
    atoms,
    bonds,
    formalCharges,
    aromatic,
    totalCharge: formalCharges.reduce((sum, q) => sum + q, 0),
    missing,
    missingHeavyAtoms,
    unexpected,
  };
}

/**
 * Hydrogens a heavy atom needs, from its bond orders and formal charge.
 *
 * The standard valence count, which is what every typing scheme needs before
 * it can place hydrogens: sum the bond orders, compare with the element's
 * neutral valence, adjust for the formal charge. Aromatic bonds count as 1.5
 * and the total is rounded, which is the usual convention and is exact for the
 * alternating Kekulé forms the CCD stores.
 */
export function implicitHydrogens(
  element: string,
  bondOrders: readonly BondOrder[],
  formalCharge: number,
): number {
  const valences: Readonly<Record<string, number>> = {
    H: 1, B: 3, C: 4, N: 3, O: 2, F: 1, SI: 4, P: 3, S: 2, CL: 1, BR: 1, I: 1,
  };
  const neutral = valences[element.toUpperCase()];
  if (neutral === undefined) return 0;
  let used = 0;
  for (const order of bondOrders) {
    used += order === "SING" ? 1 : order === "DOUB" ? 2 : order === "TRIP" ? 3 : 1.5;
  }
  // Nitrogen and phosphorus gain a valence when positively charged; oxygen and
  // the halogens lose one when negative. `+ formalCharge` captures both for the
  // elements that matter here.
  const target = neutral + (element.toUpperCase() === "C" ? 0 : formalCharge);
  return Math.max(0, Math.round(target - used));
}
