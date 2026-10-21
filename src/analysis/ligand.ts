/**
 * The ligand analyses: cells 33 to 37 of the notebook.
 *
 * Ligand RMSD, the pocket that surrounds it, distances to named residues, and
 * pulling one frame out as a structure. `docs/NOTEBOOK_PARITY.md` maps each to
 * the call it replaces.
 *
 * 🔴 **The ligand is tracked by atom index, not by residue name.** The notebook
 * selects `resname UNK`, which works because its own preparation names every
 * ligand `UNK` — and fails silently on a system prepared anywhere else, where
 * the residue keeps its PDB code (`JZ4`, `BNZ`). Selecting nothing gives an
 * empty group and MDAnalysis then reports an RMSD of zero rather than an error.
 * So the selection happens once, explicitly, and everything downstream takes
 * indices.
 *
 * 🔴 **The notebook's ligand RMSD is not what most people read it as.**
 * `MDAnalysis.analysis.rms.RMSD(select=...)` superimposes the *selection* on
 * the reference before measuring, so the number is the ligand's internal
 * conformational change and says nothing about whether it moved in the pocket.
 * A ligand that slid two nanometres while staying rigid reports ~0. Measured on
 * 3HTB: the ligand RMSD stays under 0.032 nm across 60 frames while its centre
 * of mass wanders considerably further. Both quantities are provided here,
 * named for what they are, and `docs/DEVIATIONS.md` records it.
 */

import type { PdbAtom, PdbResidue, PdbTopology } from "./pdb.js";
import { rmsdSeries, selectFrame, type Trajectory } from "./protein.js";

/**
 * Standard atomic masses, matching the table MDAnalysis guesses from.
 *
 * Centre of mass is mass-weighted, so these have to agree with whatever the
 * reference used or the comparison drifts. Checked against the masses
 * MDAnalysis actually used on 3HTB (12.011 for carbon, 15.999 for oxygen),
 * which `tools/dump_ligand_analyses.py` records for exactly this reason.
 */
const MASSES: Readonly<Record<string, number>> = {
  H: 1.008, C: 12.011, N: 14.007, O: 15.999, F: 18.998, NA: 22.990,
  MG: 24.305, P: 30.974, S: 32.06, CL: 35.45, K: 39.098, CA: 40.078,
  FE: 55.845, ZN: 65.38, BR: 79.904, I: 126.904,
};

export function massOf(element: string): number {
  return MASSES[element.toUpperCase()] ?? 0;
}

/**
 * What counts as protein, listed positively.
 *
 * 🔴 This started as a list of what to *exclude* — water and the counter-ions —
 * and that was wrong in a way the reference caught. 3HTB's ligand is residue
 * `UNK` **numbered 1**, exactly like its `MET` 1, its first water and its first
 * sodium: a PDB residue number is unique only within a chain and an insertion
 * code, and this file has four residues numbered 1. So an exclusion list that
 * did not name `UNK` put the ligand in its own binding pocket, at zero
 * distance from itself, and the pocket came out with sixteen residues against
 * MDAnalysis's fifteen.
 *
 * Matching MDAnalysis's `protein` selection, which is a positive list of
 * recognised amino acids, removes the whole class of error: a cofactor, a
 * second ligand or a modified residue is excluded because it is not named
 * here, rather than included because nobody thought to exclude it.
 */
const PROTEIN_RESIDUES = new Set([
  "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
  "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
  // The protonation and linkage variants the force fields use.
  "HID", "HIE", "HIP", "HSD", "HSE", "HSP", "CYX", "CYM", "ASH", "GLH", "LYN",
  "ARN", "TYM", "MSE", "HYP", "SEC", "PYL",
  // Terminal variants, which Amber names with an N or C prefix.
  "NALA", "NARG", "NASN", "NASP", "NCYS", "NGLN", "NGLU", "NGLY", "NHIS",
  "NILE", "NLEU", "NLYS", "NMET", "NPHE", "NPRO", "NSER", "NTHR", "NTRP",
  "NTYR", "NVAL", "CALA", "CARG", "CASN", "CASP", "CCYS", "CGLN", "CGLU",
  "CGLY", "CHIS", "CILE", "CLEU", "CLYS", "CMET", "CPHE", "CPRO", "CSER",
  "CTHR", "CTRP", "CTYR", "CVAL", "ACE", "NME", "NHE",
]);

const isProtein = (residue: PdbResidue): boolean =>
  PROTEIN_RESIDUES.has(residue.name.toUpperCase());

/**
 * The ligand's heavy atoms: the notebook's `resname X and not (name H*)`.
 *
 * Hydrogens are excluded because the notebook excludes them, and the reason is
 * sound: a deposited ligand often has none, so including them would make the
 * RMSD depend on whether the preparation added any.
 */
export function ligandHeavyAtoms(
  topology: PdbTopology,
  residueName: string,
): readonly number[] {
  const wanted = residueName.toUpperCase();
  const found: number[] = [];
  for (const residue of topology.residues) {
    if (residue.name.toUpperCase() !== wanted) continue;
    for (const atom of residue.atoms) {
      const entry = topology.atoms[atom] as PdbAtom;
      // `not (name H*)` is a *name* test in the notebook, not an element test.
      // They differ: a carbon called `HC` would be excluded by the name rule.
      // Matching the name rule is what keeps the atom count identical.
      if (entry.name.toUpperCase().startsWith("H")) continue;
      found.push(atom);
    }
  }
  return found;
}

/**
 * Ligand RMSD against one frame, superposing the ligand on itself.
 *
 * Cell 34. The same best-fit RMSD the protein analyses use, over the ligand's
 * heavy atoms — which is what makes it a conformational measure. See the note
 * at the top of this file.
 */
export function ligandRmsd(
  trajectory: Trajectory,
  ligand: readonly number[],
  referenceFrame = 0,
): Float64Array {
  return rmsdSeries(trajectory, ligand, referenceFrame);
}

/** Mass-weighted centre of mass of a selection, per frame, nm. */
export function centreOfMassSeries(
  trajectory: Trajectory,
  topology: PdbTopology,
  selection: readonly number[],
): Float64Array {
  const masses = selection.map((atom) =>
    massOf((topology.atoms[atom] as PdbAtom).element),
  );
  const total = masses.reduce((sum, m) => sum + m, 0);
  const out = new Float64Array(trajectory.frameCount * 3);
  for (let frame = 0; frame < trajectory.frameCount; frame++) {
    const coordinates = selectFrame(
      trajectory.positions,
      trajectory.particleCount,
      frame,
      selection,
    );
    let x = 0;
    let y = 0;
    let z = 0;
    for (const [slot, mass] of masses.entries()) {
      x += mass * (coordinates[slot * 3] as number);
      y += mass * (coordinates[slot * 3 + 1] as number);
      z += mass * (coordinates[slot * 3 + 2] as number);
    }
    out[frame * 3] = x / total;
    out[frame * 3 + 1] = y / total;
    out[frame * 3 + 2] = z / total;
  }
  return out;
}

/**
 * The residues lining the pocket: any protein residue with an atom within
 * `cutoff` of a ligand atom, **in one frame**.
 *
 * 🔴 Cell 36 fixes the pocket from the *first* frame and then holds it for the
 * whole trajectory. That is a choice with consequences — a residue that swings
 * into contact later never appears — and it is reproduced rather than improved,
 * because the distances that follow are only comparable with the notebook's if
 * the residue set is.
 *
 * Returned as residue *numbers* (the PDB's own `resid`), sorted, because that
 * is what the notebook prints and what a user types into cell 37.
 */
export function pocketResidues(
  topology: PdbTopology,
  ligand: readonly number[],
  frame: Float32Array | Float64Array,
  particleCount: number,
  frameIndex: number,
  cutoff: number,
): readonly number[] {
  const base = frameIndex * particleCount * 3;
  const ligandPoints = ligand.map((atom) => [
    frame[base + atom * 3] as number,
    frame[base + atom * 3 + 1] as number,
    frame[base + atom * 3 + 2] as number,
  ] as const);
  const squared = cutoff * cutoff;
  const found = new Set<number>();
  for (const residue of topology.residues) {
    if (!isProtein(residue)) continue;
    let near = false;
    for (const atom of residue.atoms) {
      const x = frame[base + atom * 3] as number;
      const y = frame[base + atom * 3 + 1] as number;
      const z = frame[base + atom * 3 + 2] as number;
      for (const point of ligandPoints) {
        const dx = x - point[0];
        const dy = y - point[1];
        const dz = z - point[2];
        if (dx * dx + dy * dy + dz * dz <= squared) {
          near = true;
          break;
        }
      }
      if (near) break;
    }
    if (near) found.add(Number(residue.id));
  }
  return [...found].sort((a, b) => a - b);
}

/** Atoms of the protein residues with these numbers. */
export function residueAtoms(
  topology: PdbTopology,
  residueNumbers: readonly number[],
): ReadonlyMap<number, readonly number[]> {
  const wanted = new Set(residueNumbers);
  const out = new Map<number, number[]>();
  for (const residue of topology.residues) {
    if (!isProtein(residue)) continue;
    const number = Number(residue.id);
    if (!wanted.has(number)) continue;
    const existing = out.get(number);
    if (existing === undefined) out.set(number, [...residue.atoms]);
    else existing.push(...residue.atoms);
  }
  return out;
}

export interface DistanceSeries {
  readonly residue: number;
  /** nm, one per frame. */
  readonly distances: Float64Array;
  readonly mean: number;
  /** Population standard deviation, which is what the notebook prints. */
  readonly standardDeviation: number;
}

/**
 * Centre-of-mass distance from the ligand to each residue, per frame.
 *
 * Cells 36 and 37. Both centres are mass-weighted, which is what
 * `center_of_mass` means on either side.
 */
export function residueDistances(
  trajectory: Trajectory,
  topology: PdbTopology,
  ligand: readonly number[],
  residueNumbers: readonly number[],
): readonly DistanceSeries[] {
  const ligandCentre = centreOfMassSeries(trajectory, topology, ligand);
  const groups = residueAtoms(topology, residueNumbers);
  const out: DistanceSeries[] = [];
  for (const residue of [...residueNumbers].sort((a, b) => a - b)) {
    const atoms = groups.get(residue);
    if (atoms === undefined) continue;
    const centre = centreOfMassSeries(trajectory, topology, atoms);
    const distances = new Float64Array(trajectory.frameCount);
    for (let frame = 0; frame < trajectory.frameCount; frame++) {
      distances[frame] = Math.hypot(
        (centre[frame * 3] as number) - (ligandCentre[frame * 3] as number),
        (centre[frame * 3 + 1] as number) - (ligandCentre[frame * 3 + 1] as number),
        (centre[frame * 3 + 2] as number) - (ligandCentre[frame * 3 + 2] as number),
      );
    }
    let mean = 0;
    for (const value of distances) mean += value;
    mean /= distances.length;
    let variance = 0;
    for (const value of distances) variance += (value - mean) ** 2;
    out.push({
      residue,
      distances,
      mean,
      standardDeviation: Math.sqrt(variance / distances.length),
    });
  }
  return out;
}

/**
 * One frame as a topology, with the solvent dropped.
 *
 * Cell 33: `protein_ligand_{frame}.pdb`, no water and no ions. Returns a
 * topology rather than text so the caller can write it with `writePdb` or
 * hand it to anything else that takes one.
 */
export function extractFrame(
  topology: PdbTopology,
  trajectory: Trajectory,
  frameIndex: number,
  options: { readonly keepSolvent?: boolean } = {},
): PdbTopology {
  // Everything that is not solvent: the protein *and* the ligand, which is the
  // point of the file. `isProtein` would drop the ligand.
  const SOLVENT = new Set(["HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "ZN",
    "MN", "FE", "BR", "IOD", "LI", "RB", "CS", "CA", "SO4", "PO4"]);
  const keep = (residue: PdbResidue): boolean =>
    options.keepSolvent === true || !SOLVENT.has(residue.name.toUpperCase());

  const base = frameIndex * trajectory.particleCount * 3;
  const atoms: PdbAtom[] = [];
  const residues: PdbResidue[] = [];
  const positions: number[] = [];
  const remap = new Map<number, number>();
  const residueRemap = new Map<number, number>();

  for (const [index, residue] of topology.residues.entries()) {
    if (!keep(residue)) continue;
    const kept: number[] = [];
    for (const atom of residue.atoms) {
      remap.set(atom, atoms.length);
      kept.push(atoms.length);
      atoms.push(topology.atoms[atom] as PdbAtom);
      positions.push(
        trajectory.positions[base + atom * 3] as number,
        trajectory.positions[base + atom * 3 + 1] as number,
        trajectory.positions[base + atom * 3 + 2] as number,
      );
    }
    residueRemap.set(index, residues.length);
    residues.push({ ...residue, atoms: kept });
  }
  const chains = topology.chains
    .map((chain) => ({
      id: chain.id,
      residues: chain.residues
        .map((r) => residueRemap.get(r))
        .filter((r): r is number => r !== undefined),
    }))
    .filter((chain) => chain.residues.length > 0);

  return {
    atoms,
    residues,
    chains,
    box: topology.box,
    conect: [],
    positions: Float64Array.from(positions),
  };
}
