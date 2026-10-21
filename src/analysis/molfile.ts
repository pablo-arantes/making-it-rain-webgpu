/**
 * A minimal MDL MOL/SDF reader, for a ligand's chemistry.
 *
 * The interaction fingerprint needs per-atom aromaticity, valence and
 * neighbours, and a PDB carries none of it. An SDF does — it is what the
 * notebook's own preparation writes, and what a user already has if they
 * parameterised the ligand.
 *
 * Only V2000, and only the counts, atom and bond blocks. The format has far
 * more in it; everything not needed is skipped rather than half-read.
 */

export interface MolFile {
  readonly elements: readonly string[];
  readonly formalCharges: readonly number[];
  /** `[a, b, order]` with zero-based indices; order 4 means the file said aromatic. */
  readonly bonds: readonly (readonly [number, number, number])[];
}

export class MolParseError extends Error {
  constructor(detail: string) {
    super(`this does not look like a V2000 MOL or SDF: ${detail}`);
    this.name = "MolParseError";
  }
}

/**
 * The charge column of the atom block, which is coded rather than signed.
 *
 * 0 means neutral, and 1 to 7 mean +3, +2, +1, doublet, −1, −2, −3 in that
 * order. Reading it as a number would make a −1 look like a +5.
 */
const CHARGE_CODES: Readonly<Record<number, number>> = {
  0: 0, 1: 3, 2: 2, 3: 1, 4: 0, 5: -1, 6: -2, 7: -3,
};

export function readMolV2000(text: string): MolFile {
  const lines = text.split(/\r?\n/);
  if (lines.length < 4) throw new MolParseError(`only ${lines.length} lines`);
  const counts = lines[3] as string;
  const atomCount = Number(counts.slice(0, 3).trim());
  const bondCount = Number(counts.slice(3, 6).trim());
  if (!Number.isInteger(atomCount) || !Number.isInteger(bondCount)) {
    throw new MolParseError(`the counts line reads "${counts.trim()}"`);
  }
  if (lines.length < 4 + atomCount + bondCount) {
    throw new MolParseError(
      `it declares ${atomCount} atoms and ${bondCount} bonds but has ` +
        `${lines.length - 4} lines after the counts line`,
    );
  }

  const elements: string[] = [];
  const formalCharges: number[] = [];
  for (let i = 0; i < atomCount; i++) {
    const line = lines[4 + i] as string;
    // Columns are fixed: x, y, z then the symbol at 31-34.
    elements.push(line.slice(31, 34).trim());
    formalCharges.push(CHARGE_CODES[Number(line.slice(36, 39).trim()) || 0] ?? 0);
  }

  const bonds: [number, number, number][] = [];
  for (let i = 0; i < bondCount; i++) {
    const line = lines[4 + atomCount + i] as string;
    const a = Number(line.slice(0, 3).trim()) - 1;
    const b = Number(line.slice(3, 6).trim()) - 1;
    const order = Number(line.slice(6, 9).trim());
    if (a < 0 || b < 0 || a >= atomCount || b >= atomCount) {
      throw new MolParseError(`bond ${i + 1} names atoms ${a + 1} and ${b + 1}`);
    }
    bonds.push([a, b, order]);
  }

  // `M  CHG` overrides the atom block's charge column, and when present it is
  // the authority -- RDKit writes charges there rather than in the columns.
  for (const line of lines.slice(4 + atomCount + bondCount)) {
    if (!line.startsWith("M  CHG")) continue;
    const fields = line.slice(6).trim().split(/\s+/).map(Number);
    const pairs = fields[0] ?? 0;
    for (let i = 0; i < pairs; i++) {
      const atom = (fields[1 + i * 2] ?? 0) - 1;
      const charge = fields[2 + i * 2] ?? 0;
      if (atom >= 0 && atom < atomCount) formalCharges[atom] = charge;
    }
  }

  return { elements, formalCharges, bonds };
}

/**
 * Which atoms sit in an aromatic ring.
 *
 * 🔴 Perceived, because the file usually does not say. RDKit writes Kekulé
 * bonds — alternating single and double — and leaves aromaticity to be worked
 * out, so an SDF of benzene has three single and three double bonds and no
 * aromatic flag anywhere.
 *
 * The rule here: a ring of five or six atoms, all of them C, N, O or S, in
 * which **every atom has exactly one double bond inside the ring**, or whose
 * bonds the file explicitly marked aromatic (order 4). That is Hückel for the
 * cases a drug-like ligand actually presents, and it is deliberately not a
 * general aromaticity model — a fused system where one ring borrows electrons
 * from another, or a five-ring aromatic through a lone pair like furan, is not
 * detected.
 *
 * It matters for one atom of JZ4 and that atom decides the answer. ProLIF's
 * hydrophobic pattern excludes a carbon bonded to oxygen, *except* through its
 * aromatic-carbon branch — so the phenol's C–OH carbon is hydrophobic only if
 * the ring is known to be aromatic. Without this, eight of the nine carbons
 * come out hydrophobic instead of nine.
 */
export function perceiveAromaticAtoms(mol: MolFile): readonly boolean[] {
  const n = mol.elements.length;
  const neighbours: number[][] = Array.from({ length: n }, () => []);
  const orderOf = new Map<string, number>();
  for (const [a, b, order] of mol.bonds) {
    (neighbours[a] as number[]).push(b);
    (neighbours[b] as number[]).push(a);
    orderOf.set(`${Math.min(a, b)},${Math.max(a, b)}`, order);
  }
  const order = (a: number, b: number): number =>
    orderOf.get(`${Math.min(a, b)},${Math.max(a, b)}`) ?? 0;

  const aromatic = new Array<boolean>(n).fill(false);
  const allowed = new Set(["C", "N", "O", "S"]);

  // Every cycle of length 5 or 6, by depth-first walk. Small molecules only --
  // this is exponential in principle and trivial at twenty atoms.
  const seen = new Set<string>();
  const walk = (start: number, current: number, path: number[]): void => {
    if (path.length > 6) return;
    for (const next of neighbours[current] as number[]) {
      if (next === start && path.length >= 5) {
        const ring = [...path].sort((x, y) => x - y);
        const key = ring.join(",");
        if (seen.has(key)) continue;
        seen.add(key);
        if (!path.every((atom) => allowed.has((mol.elements[atom] as string).toUpperCase()))) {
          continue;
        }
        const explicit = path.every((atom, index) => {
          const other = path[(index + 1) % path.length] as number;
          return order(atom, other) === 4;
        });
        const alternating = path.every((atom) => {
          let doubles = 0;
          for (const other of path) {
            if (other !== atom && order(atom, other) === 2) doubles++;
          }
          return doubles === 1;
        });
        if (explicit || alternating) {
          for (const atom of path) aromatic[atom] = true;
        }
        continue;
      }
      if (path.includes(next)) continue;
      walk(start, next, [...path, next]);
    }
  };
  for (let atom = 0; atom < n; atom++) walk(atom, atom, [atom]);
  return aromatic;
}
