/**
 * Rewriting a PDB with new coordinates.
 *
 * Each stage writes a PDB of its final frame, which is what the analysis stage
 * uses as a topology -- the notebook's `Equilibrated_PDB` field. The engine does
 * not model a topology: atom names, residues, chains and elements come from the
 * file the user uploaded, and this substitutes coordinates into it rather than
 * generating one. That is the whole design, and it means a rewritten file keeps
 * every record the original had, including the ones nothing here understands.
 *
 * 🔴 A PDB is **not** a checkpoint. Coordinates are `Real(8.3)` in ångström, so
 * 0.001 Å -- 1e-4 nm -- and a position is stored to about a thousandth of the
 * precision the constraint tolerance needs. Restarting from one would violate
 * every constrained bond by a factor of a thousand more than the solver allows.
 * The `.rst` (a serialised `State`) is the checkpoint; this is for topology and
 * for looking at.
 *
 * Column positions are fixed, not whitespace-delimited, which is the trap in
 * this format: an atom named `CA` and one named `1HB` are distinguished by
 * which column the text starts in, and a four-character name fills the field so
 * there is no space before the residue. Nothing is re-flowed here -- only the
 * three coordinate fields and `CRYST1` are touched, byte range by byte range.
 */

import type { Vec3 } from "../engine/system.js";

/** Nanometres to ångström, which is what a PDB holds. */
const TO_ANGSTROM = 10;

export class PdbAtomCountMismatch extends Error {
  constructor(readonly inFile: number, readonly inSystem: number) {
    super(
      `the PDB has ${inFile} ATOM/HETATM records and the system has ` +
        `${inSystem} particles. Coordinates are substituted by position, so ` +
        "the two have to describe the same atoms in the same order -- a PDB " +
        "from a different stage of preparation, or one with hydrogens the " +
        "system does not have, would silently put every coordinate on the " +
        "wrong atom.",
    );
    this.name = "PdbAtomCountMismatch";
  }
}

/** Right-justify a fixed-point number into `width` columns, as PDB requires. */
function fixed(value: number, width: number, decimals: number): string {
  const text = value.toFixed(decimals);
  if (text.length > width) {
    throw new Error(
      `${text} does not fit the ${width}-column PDB field it belongs in. A ` +
        "coordinate beyond +/-9999.999 A cannot be written: the format has no " +
        "wider field, and silently truncating it would move the atom.",
    );
  }
  return text.padStart(width, " ");
}

export const isAtomRecord = (line: string): boolean =>
  line.startsWith("ATOM") || line.startsWith("HETATM");

/** How many atoms a PDB describes. */
export function countAtoms(pdb: string): number {
  let count = 0;
  for (const line of pdb.split("\n")) if (isAtomRecord(line)) count++;
  return count;
}

export interface RewriteOptions {
  /** nm. Replaces `CRYST1` when given; a run's box changes under a barostat. */
  readonly box?: Vec3;
  /** Prepended as a REMARK, so a file says where it came from. */
  readonly remark?: string;
}

/**
 * Substitute coordinates into a PDB, keeping everything else.
 *
 * `positions` is nm, xyz interleaved, in the order the file's ATOM records
 * appear.
 */
export function rewritePdb(
  pdb: string,
  positions: Float64Array,
  options: RewriteOptions = {},
): string {
  const atoms = positions.length / 3;
  const inFile = countAtoms(pdb);
  if (inFile !== atoms) throw new PdbAtomCountMismatch(inFile, atoms);

  const lines = pdb.split("\n");
  const out: string[] = [];
  if (options.remark !== undefined) {
    // REMARK 1 is free text; OpenMM writes its own provenance there too.
    out.push(`REMARK   1 ${options.remark}`.padEnd(80, " ").slice(0, 80));
  }

  let index = 0;
  for (const line of lines) {
    if (isAtomRecord(line)) {
      const x = fixed((positions[index * 3] as number) * TO_ANGSTROM, 8, 3);
      const y = fixed((positions[index * 3 + 1] as number) * TO_ANGSTROM, 8, 3);
      const z = fixed((positions[index * 3 + 2] as number) * TO_ANGSTROM, 8, 3);
      // Columns 31-54, one-based, which is 30..54 as a slice. Everything
      // outside that range is carried over untouched -- the atom name's exact
      // column matters, and so does whatever the original put after column 54.
      const padded = line.padEnd(54, " ");
      out.push(padded.slice(0, 30) + x + y + z + padded.slice(54));
      index++;
      continue;
    }
    if (line.startsWith("CRYST1") && options.box !== undefined) {
      const [a, b, c] = options.box as unknown as [number, number, number];
      // Right angles: the engine refuses triclinic boxes upstream.
      out.push(
        "CRYST1" +
          fixed(a * TO_ANGSTROM, 9, 3) +
          fixed(b * TO_ANGSTROM, 9, 3) +
          fixed(c * TO_ANGSTROM, 9, 3) +
          fixed(90, 7, 2) +
          fixed(90, 7, 2) +
          fixed(90, 7, 2) +
          " P 1           1 ",
      );
      continue;
    }
    out.push(line);
  }
  return out.join("\n");
}
