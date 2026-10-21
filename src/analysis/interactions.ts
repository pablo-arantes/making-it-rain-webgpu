/**
 * The protein–ligand interaction fingerprint: the notebook's cell 29.
 *
 * ProLIF's aggregate network, thresholded on how often each interaction
 * appears. Two of its interaction types are implemented here and the rest are
 * not; which, and why, is stated at the bottom.
 *
 * 🔴 **Getting a meaningful reference out of ProLIF takes two things that are
 * easy to miss, and without either it reports almost nothing while appearing
 * to work.** MDAnalysis does not infer bonds from a PDB, so the protein reaches
 * RDKit as disconnected atoms and no SMARTS pattern matches: every
 * chemistry-dependent interaction vanishes and only `VdWContact` survives,
 * because it needs nothing but elements and distances. And a ligand from a PDB
 * has no bond orders, so its aromaticity and valence are unknown. Measured on
 * 3HTB: one interaction before both fixes, twenty-six after. A parity test
 * against the first number would have been meaningless, which is why
 * `tools/dump_interactions.py` does both.
 *
 * **Chemistry is an input, not a derivation.** Everything here takes per-atom
 * element, aromaticity, formal charge, hydrogen count and neighbours. A PDB
 * does not carry that for a ligand, so the caller supplies it — from the CCD
 * (`src/prep/ccd.ts`) or an SDF. That is a real constraint rather than a
 * shortcut: nothing can decide whether a ring is aromatic from coordinates.
 */

/** ProLIF's `VDWRADII`, in nm. Its own table is in ångström. */
export const VDW_RADII: Readonly<Record<string, number>> = {
  H: 0.11, HE: 0.14, LI: 0.182, BE: 0.153, B: 0.192, C: 0.17, N: 0.155,
  O: 0.152, F: 0.147, NE: 0.154, NA: 0.227, MG: 0.173, AL: 0.184, SI: 0.21,
  P: 0.18, S: 0.18, CL: 0.175, AR: 0.188, K: 0.275, CA: 0.231, BR: 0.185,
  I: 0.198, ZN: 0.139, FE: 0.194, MN: 0.205, CU: 0.14, SE: 0.19,
};

/** ProLIF's defaults, converted to nm. */
export const HYDROPHOBIC_DISTANCE = 0.45;
export const VDW_TOLERANCE = 0;

export interface AtomChemistry {
  readonly element: string;
  readonly aromatic: boolean;
  readonly formalCharge: number;
  /** Hydrogens on this atom, counting explicit neighbours. */
  readonly hydrogens: number;
  /** Elements of every bonded neighbour, hydrogens included. */
  readonly neighbours: readonly string[];
  /** Summed bond orders, which the thioether test needs. */
  readonly valence: number;
}

const ELECTRONEGATIVE = new Set(["N", "O", "F"]);

/**
 * ProLIF's hydrophobic test, as a rule rather than a SMARTS match.
 *
 * The pattern is
 * `[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]`,
 * which reads: an aromatic carbon or sulfur, or a bromine or iodine, or a
 * thioether sulfur, or a carbon of degree 3 or 4 that is **not** bonded to
 * nitrogen, oxygen or fluorine and is **not** a quaternary sp3 carbon — all
 * with a formal charge of zero.
 *
 * Evaluating it directly avoids a SMARTS engine, which would mean RDKit in
 * WebAssembly for this one predicate. Verified against ProLIF's own
 * substructure match on JZ4: both select exactly the nine carbons and not the
 * hydroxyl oxygen.
 */
export function isHydrophobic(atom: AtomChemistry): boolean {
  if (atom.formalCharge !== 0) return false;
  const element = atom.element.toUpperCase();
  if (atom.aromatic && (element === "C" || element === "S")) return true;
  if (element === "BR" || element === "I") return true;
  // `S&H0&v2`: a thioether, not a thiol and not a sulfone.
  if (element === "S" && atom.hydrogens === 0 && atom.valence === 2) return true;
  if (element !== "C") return false;

  const degree = atom.neighbours.length;
  if (degree !== 3 && degree !== 4) return false;
  for (const neighbour of atom.neighbours) {
    if (ELECTRONEGATIVE.has(neighbour.toUpperCase())) return false;
  }
  // `!$([#6X4H0])`: a carbon with four heavy neighbours and no hydrogen.
  if (degree === 4 && atom.hydrogens === 0) return false;
  return true;
}

export interface Contact {
  /** Residue number, as the PDB writes it. */
  readonly residue: number;
  readonly interaction: "Hydrophobic" | "VdWContact";
}

export interface FingerprintOptions {
  readonly hydrophobicDistance?: number;
  readonly vdwTolerance?: number;
}

export interface FingerprintResult {
  /** `${residue}|${interaction}` -> one boolean per frame. */
  readonly frames: ReadonlyMap<string, readonly boolean[]>;
  /** The same keys, with the fraction of frames the interaction appears in. */
  readonly frequency: ReadonlyMap<string, number>;
  readonly frameCount: number;
}

interface Group {
  readonly residue: number;
  readonly atoms: readonly number[];
}

/**
 * Which interactions each residue makes with the ligand, frame by frame.
 *
 * A residue scores an interaction if **any** pair of its atoms and the
 * ligand's satisfies the criterion in that frame, which is ProLIF's rule: the
 * fingerprint is per residue, not per atom pair.
 *
 * The per-frame booleans are returned alongside the frequencies because a
 * frequency alone can hide compensating errors — two implementations can agree
 * on an average over sixty frames while disagreeing about which frames.
 */
export function interactionFingerprint(
  positions: Float32Array | Float64Array,
  particleCount: number,
  frameCount: number,
  ligand: readonly number[],
  groups: readonly Group[],
  chemistry: ReadonlyMap<number, AtomChemistry>,
  options: FingerprintOptions = {},
): FingerprintResult {
  const hydrophobicCutoff = options.hydrophobicDistance ?? HYDROPHOBIC_DISTANCE;
  const tolerance = options.vdwTolerance ?? VDW_TOLERANCE;

  const radius = (atom: number): number => {
    const element = chemistry.get(atom)?.element.toUpperCase();
    return element === undefined ? 0 : (VDW_RADII[element] ?? 0);
  };
  const hydrophobic = (atom: number): boolean => {
    const entry = chemistry.get(atom);
    return entry !== undefined && isHydrophobic(entry);
  };

  const ligandHydrophobic = ligand.filter(hydrophobic);
  const frames = new Map<string, boolean[]>();
  const record = (key: string, frame: number, value: boolean): void => {
    let series = frames.get(key);
    if (series === undefined) {
      series = new Array<boolean>(frameCount).fill(false);
      frames.set(key, series);
    }
    if (value) series[frame] = true;
  };

  for (let frame = 0; frame < frameCount; frame++) {
    const base = frame * particleCount * 3;
    const at = (atom: number): readonly [number, number, number] => [
      positions[base + atom * 3] as number,
      positions[base + atom * 3 + 1] as number,
      positions[base + atom * 3 + 2] as number,
    ];
    const ligandPoints = ligand.map(at);
    const hydrophobicPoints = ligandHydrophobic.map(at);

    for (const group of groups) {
      let anyHydrophobic = false;
      let anyVdw = false;
      for (const atom of group.atoms) {
        const point = at(atom);
        const atomRadius = radius(atom);
        if (!anyVdw && atomRadius > 0) {
          for (const [slot, other] of ligandPoints.entries()) {
            const limit = atomRadius + radius(ligand[slot] as number) + tolerance;
            if (limit <= 0) continue;
            const dx = point[0] - other[0];
            const dy = point[1] - other[1];
            const dz = point[2] - other[2];
            if (dx * dx + dy * dy + dz * dz <= limit * limit) {
              anyVdw = true;
              break;
            }
          }
        }
        if (!anyHydrophobic && hydrophobic(atom)) {
          const limit = hydrophobicCutoff * hydrophobicCutoff;
          for (const other of hydrophobicPoints) {
            const dx = point[0] - other[0];
            const dy = point[1] - other[1];
            const dz = point[2] - other[2];
            if (dx * dx + dy * dy + dz * dz <= limit) {
              anyHydrophobic = true;
              break;
            }
          }
        }
        if (anyHydrophobic && anyVdw) break;
      }
      if (anyHydrophobic) record(`${group.residue}|Hydrophobic`, frame, true);
      if (anyVdw) record(`${group.residue}|VdWContact`, frame, true);
    }
  }

  const frequency = new Map<string, number>();
  for (const [key, series] of frames) {
    frequency.set(key, series.filter(Boolean).length / frameCount);
  }
  return { frames, frequency, frameCount };
}

/**
 * What is **not** implemented, and why it is listed rather than approximated.
 *
 * ProLIF's default set is Anionic, CationPi, Cationic, HBAcceptor, HBDonor,
 * Hydrophobic, PiCation, PiStacking and VdWContact. Two are here.
 *
 * The hydrogen bonds and the charged and π interactions each need more than a
 * distance: a donor–hydrogen–acceptor angle, a ring centroid and normal, or a
 * formal-charge assignment over a whole functional group. Each is a few lines
 * of geometry on top of atom *typing* that a PDB does not provide — which is
 * the same constraint as above, not a different one. `HBDonor` appears in the
 * 3HTB reference, so this fingerprint is a subset of it by construction and the
 * test says so rather than comparing only the keys it happens to produce.
 */
export const IMPLEMENTED_INTERACTIONS = ["Hydrophobic", "VdWContact"] as const;

/**
 * Aromatic ring atoms of the standard residues, by name.
 *
 * 🔴 A table rather than perception, because a PDB gives the protein no bond
 * orders either — `inferBonds` produces connectivity and nothing more, so the
 * alternating-bond rule in `perceiveAromaticAtoms` has nothing to work with.
 *
 * It matters for exactly the atoms whose aromaticity changes the answer.
 * ProLIF's hydrophobic pattern excludes a carbon bonded to nitrogen or oxygen
 * *except* through its aromatic branch, so phenylalanine's ring needs no help —
 * every ring carbon has only carbon and hydrogen neighbours — while
 * tyrosine's CZ, histidine's CD2 and CE1 and tryptophan's CD1 and CE2 are
 * hydrophobic only if the ring is known to be aromatic.
 */
const AROMATIC_ATOMS: Readonly<Record<string, readonly string[]>> = {
  PHE: ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
  TYR: ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
  TRP: ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
  HIS: ["CG", "ND1", "CD2", "CE1", "NE2"],
  HID: ["CG", "ND1", "CD2", "CE1", "NE2"],
  HIE: ["CG", "ND1", "CD2", "CE1", "NE2"],
  HIP: ["CG", "ND1", "CD2", "CE1", "NE2"],
};

/**
 * Per-atom chemistry for a protein, from its topology and bonds.
 *
 * Formal charges are left at zero, which is right for the hydrophobic and van
 * der Waals tests here — neither reads them for a protein atom — and would not
 * be for the charged interactions this module does not implement.
 */
export function proteinChemistry(
  atoms: readonly {
    readonly name: string;
    readonly element: string;
    readonly residueName: string;
  }[],
  bonds: readonly (readonly [number, number])[],
): ReadonlyMap<number, AtomChemistry> {
  const neighbours: number[][] = atoms.map(() => []);
  for (const [a, b] of bonds) {
    (neighbours[a] as number[]).push(b);
    (neighbours[b] as number[]).push(a);
  }
  const out = new Map<number, AtomChemistry>();
  for (const [index, atom] of atoms.entries()) {
    const aromaticNames = AROMATIC_ATOMS[atom.residueName.toUpperCase()];
    const connected = neighbours[index] as number[];
    out.set(index, {
      element: atom.element,
      aromatic: aromaticNames?.includes(atom.name.toUpperCase()) ?? false,
      formalCharge: 0,
      hydrogens: connected.filter(
        (other) => (atoms[other] as { element: string }).element.toUpperCase() === "H",
      ).length,
      neighbours: connected.map(
        (other) => (atoms[other] as { element: string }).element,
      ),
      // Single bonds throughout: a protein's connectivity is all this has, and
      // the only test that reads valence is the thioether one, where a
      // cysteine or methionine sulfur has two single bonds either way.
      valence: connected.length,
    });
  }
  return out;
}
