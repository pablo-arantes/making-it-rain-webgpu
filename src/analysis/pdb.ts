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

/**
 * Ångström to nanometres, on the way in.
 *
 * 🔴 Multiplying by 0.1 and dividing by 10 are **different doubles**, and this
 * is the one that matches OpenMM. Its unit system converts by multiplying by
 * the factor 0.1, so a `CRYST1` of 30.806 Å becomes 3.0806000000000004 nm,
 * while `30.806 / 10` is 3.0806000000000000. One ulp, 4.4e-16 nm, far below
 * anything physical -- and exactly enough to make a box comparison against a
 * serialised `System` fail for a reason that looks like nothing.
 *
 * Keeping the two conversions bit-identical is what lets
 * test/prep-system.test.ts compare with no tolerance at all, which is where
 * its value comes from.
 */
const FROM_ANGSTROM = 0.1;

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
      // 🔴 Right angles, because `box` is a diagonal and cannot say otherwise.
      // The engine runs a skewed cell now, so for one of those this writes a
      // rectangular `CRYST1` of the right volume and the wrong shape. Nothing
      // downstream of a *run* reads it -- the box comes from `state.xml` -- but
      // a reader of the PDB alone would be misled, which is deviation 24.
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

/**
 * Reading a PDB into a topology.
 *
 * Fixed-width, not whitespace-delimited: the atom name occupies columns 13-16
 * and *where in that field the text starts carries meaning*. A one- or
 * two-letter element is right-justified from column 14, so ` CA ` is an alpha
 * carbon and `CA  ` is a calcium ion. Splitting on spaces loses that
 * distinction and silently turns an ion into a backbone atom.
 */

export interface PdbAtom {
  readonly serial: number;
  /** Trimmed, but see `rawName` -- the untrimmed field is what disambiguates. */
  readonly name: string;
  readonly rawName: string;
  readonly altLoc: string;
  readonly residueName: string;
  readonly chainId: string;
  /** The residue sequence number and insertion code, as written. */
  readonly residueId: string;
  /** nm, converted from the file's ångström. */
  readonly position: readonly [number, number, number];
  /** From columns 77-78 when present, else inferred from the name. */
  readonly element: string;
  readonly hetatm: boolean;
}

export interface PdbResidue {
  readonly name: string;
  readonly id: string;
  readonly chainId: string;
  /** Indices into the topology's atom list. */
  readonly atoms: readonly number[];
}

export interface PdbTopology {
  readonly atoms: readonly PdbAtom[];
  readonly residues: readonly PdbResidue[];
  /** One entry per chain, holding indices into `residues`. */
  readonly chains: readonly { readonly id: string; readonly residues: readonly number[] }[];
  /** nm, from CRYST1, when the file has one. */
  readonly box: readonly [number, number, number] | null;
  /** Bonds the file asserted with CONECT, as atom index pairs. */
  readonly conect: readonly (readonly [number, number])[];
  /** Positions, nm, xyz interleaved, in atom order. */
  readonly positions: Float64Array;
}

/**
 * Guess an element from an atom name when the file omits columns 77-78.
 *
 * Old files do omit them. The rule is OpenMM's: the first alphabetic character
 * of the *left-justified* name, except that a leading digit means a hydrogen
 * (`1HB`), which is the convention PDB files use for hydrogens attached to a
 * carbon with several of them.
 */
function guessElement(rawName: string): string {
  const trimmed = rawName.trim();
  if (trimmed.length === 0) return "";
  if (/^[0-9]/.test(trimmed)) return "H";
  // A two-letter element is left-justified in the four-column field, so a name
  // starting in column 13 rather than 14 is a clue -- but names are not always
  // written that carefully, and the element column is the reliable source.
  return (trimmed[0] as string).toUpperCase();
}

export class PdbParseError extends Error {
  constructor(message: string, readonly line: number) {
    super(`${message} (line ${line})`);
    this.name = "PdbParseError";
  }
}

export function readPdb(text: string): PdbTopology {
  const atoms: PdbAtom[] = [];
  // Mutable while building, and the result type is read-only: appending through
  // a cast to the read-only shape is what TypeScript objects to, correctly.
  const residues: { name: string; id: string; chainId: string; atoms: number[] }[] = [];
  const chains: { id: string; residues: number[] }[] = [];
  const conect: [number, number][] = [];
  const bySerial = new Map<number, number>();
  let box: [number, number, number] | null = null;

  let currentResidue: { key: string; index: number } | null = null;
  let currentChain: { id: string; index: number } | null = null;
  let lineNumber = 0;

  for (const line of text.split("\n")) {
    lineNumber++;
    if (line.startsWith("CRYST1")) {
      box = [
        Number(line.slice(6, 15)) * FROM_ANGSTROM,
        Number(line.slice(15, 24)) * FROM_ANGSTROM,
        Number(line.slice(24, 33)) * FROM_ANGSTROM,
      ];
      continue;
    }
    if (line.startsWith("CONECT")) {
      // Columns 7-11 then four more fields of five, one-based.
      const first = Number(line.slice(6, 11));
      for (const start of [11, 16, 21, 26]) {
        const field = line.slice(start, start + 5).trim();
        if (field === "") continue;
        conect.push([first, Number(field)]);
      }
      continue;
    }
    if (line.startsWith("TER")) {
      // A chain break: the next residue starts a new chain even if the chain id
      // is unchanged, which happens in files with two copies of one chain.
      currentChain = null;
      currentResidue = null;
      continue;
    }
    if (!isAtomRecord(line)) continue;

    const padded = line.padEnd(80, " ");
    const rawName = padded.slice(12, 16);
    const altLoc = padded.slice(16, 17).trim();
    // 🔴 Alternate locations: keeping more than one would double up atoms and
    // break every template match. The first is kept, which is what OpenMM does.
    if (altLoc !== "" && altLoc !== "A") continue;

    const residueName = padded.slice(17, 20).trim();
    const chainId = padded.slice(21, 22).trim();
    const residueId = padded.slice(22, 27).trim();
    const elementField = padded.slice(76, 78).trim();
    const serial = Number(padded.slice(6, 11));
    const position: [number, number, number] = [
      Number(padded.slice(30, 38)) * FROM_ANGSTROM,
      Number(padded.slice(38, 46)) * FROM_ANGSTROM,
      Number(padded.slice(46, 54)) * FROM_ANGSTROM,
    ];
    if (position.some((value) => !Number.isFinite(value))) {
      throw new PdbParseError(`unreadable coordinates in "${line.trim()}"`, lineNumber);
    }

    const chainKey = chainId;
    if (currentChain === null || currentChain.id !== chainKey) {
      currentChain = { id: chainKey, index: chains.length };
      chains.push({ id: chainKey, residues: [] });
    }
    const residueKey = `${chainKey}|${residueId}|${residueName}`;
    if (currentResidue === null || currentResidue.key !== residueKey) {
      currentResidue = { key: residueKey, index: residues.length };
      residues.push({ name: residueName, id: residueId, chainId, atoms: [] });
      (chains[currentChain.index] as { residues: number[] }).residues.push(
        currentResidue.index,
      );
    }

    bySerial.set(serial, atoms.length);
    (residues[currentResidue.index] as { atoms: number[] }).atoms.push(atoms.length);
    atoms.push({
      serial,
      name: rawName.trim(),
      rawName,
      altLoc,
      residueName,
      chainId,
      residueId,
      position,
      element: elementField === "" ? guessElement(rawName) : elementField,
      hetatm: line.startsWith("HETATM"),
    });
  }

  const positions = new Float64Array(atoms.length * 3);
  for (const [index, atom] of atoms.entries()) {
    positions[index * 3] = atom.position[0];
    positions[index * 3 + 1] = atom.position[1];
    positions[index * 3 + 2] = atom.position[2];
  }

  // CONECT records name serial numbers, which are not indices.
  const resolved: [number, number][] = [];
  for (const [a, b] of conect) {
    const first = bySerial.get(a);
    const second = bySerial.get(b);
    if (first === undefined || second === undefined) continue;
    if (first < second) resolved.push([first, second]);
  }

  return { atoms, residues, chains, box, conect: resolved, positions };
}

/**
 * Write a PDB from a topology, for a structure that never had a file.
 *
 * `rewritePdb` substitutes coordinates into an existing file, which is right
 * for a trajectory frame and useless for a system the browser built: a
 * solvated box has thousands of atoms no input file ever mentioned. This emits
 * the records from scratch.
 *
 * Column positions are the format's, not a guess: serial 7-11, name 13-16,
 * residue name 18-20, chain 22, residue sequence 23-26, coordinates 31-54 in
 * ångström, element 77-78. The atom name is left-padded by one space when it
 * is short enough, which is the convention that distinguishes a carbon alpha
 * `CA` from a calcium `CA` — the reader in this file relies on it, so the
 * writer has to produce it.
 *
 * `TER` goes after the last residue of each chain, and chains are written in
 * topology order so the file reads back as the same topology.
 */
export function writePdb(
  topology: PdbTopology,
  options: { readonly remark?: string } = {},
): string {
  const out: string[] = [];
  if (options.remark !== undefined) {
    out.push(`REMARK   1 ${options.remark}`.padEnd(80, " ").slice(0, 80));
  }
  if (topology.box !== null) {
    // 🔴 Right angles, for the reason at the other `CRYST1` above: a
    // `PdbTopology` carries a diagonal, so a skewed cell cannot survive this
    // file. Deviation 24.
    const [a, b, c] = topology.box;
    out.push(
      "CRYST1" +
        fixed(a * TO_ANGSTROM, 9, 3) +
        fixed(b * TO_ANGSTROM, 9, 3) +
        fixed(c * TO_ANGSTROM, 9, 3) +
        fixed(90, 7, 2) +
        fixed(90, 7, 2) +
        fixed(90, 7, 2) +
        " P 1           1",
    );
  }

  let serial = 1;
  for (const chain of topology.chains) {
    let last: { name: string; id: string } | undefined;
    for (const residueIndex of chain.residues) {
      const residue = topology.residues[residueIndex] as PdbResidue;
      for (const atom of residue.atoms) {
        const entry = topology.atoms[atom] as PdbAtom;
        // 🔴 Columns 13-14 hold the element symbol, right-justified, and that
        // is what decides where the name starts. A single-letter element gets
        // a leading space -- ` CA ` is a carbon alpha -- while a two-letter one
        // starts in column 13, so `NA  ` is a sodium. Transcribed from
        // `PDBFile.writeFile`: prepend a space only when the name is short, it
        // begins with a letter, *and* the element symbol is one character.
        //
        // Getting this wrong is the same ambiguity `readPdb` keeps `rawName`
        // for, seen from the writing side: a round trip turned a sodium `Na  `
        // into ` Na `, which reads back as the element N.
        const shift =
          entry.name.length < 4 &&
          /^[A-Za-z]/.test(entry.name) &&
          entry.element.length < 2;
        const name = shift
          ? ` ${entry.name}`.padEnd(4, " ")
          : entry.name.padEnd(4, " ").slice(0, 4);
        const x = fixed((topology.positions[atom * 3] as number) * TO_ANGSTROM, 8, 3);
        const y = fixed(
          (topology.positions[atom * 3 + 1] as number) * TO_ANGSTROM,
          8,
          3,
        );
        const z = fixed(
          (topology.positions[atom * 3 + 2] as number) * TO_ANGSTROM,
          8,
          3,
        );
        out.push(
          (entry.hetatm ? "HETATM" : "ATOM  ") +
            String(serial % 100000).padStart(5, " ") +
            " " +
            name +
            " " +
            residue.name.padStart(3, " ").slice(0, 3) +
            " " +
            (chain.id === "" ? " " : chain.id.slice(0, 1)) +
            residue.id.padStart(4, " ").slice(0, 5) +
            "    " +
            x +
            y +
            z +
            "  1.00  0.00          " +
            // As the topology holds it: OpenMM writes `Na`, not `NA`, and the
            // reader is case-insensitive, so preserving it round-trips.
            entry.element.padStart(2, " "),
        );
        serial++;
      }
      last = { name: residue.name, id: residue.id };
    }
    if (last !== undefined) {
      out.push(
        "TER   " +
          String(serial % 100000).padStart(5, " ") +
          "      " +
          last.name.padStart(3, " ").slice(0, 3) +
          " " +
          (chain.id === "" ? " " : chain.id.slice(0, 1)) +
          last.id.padStart(4, " ").slice(0, 5),
      );
      serial++;
    }
  }
  out.push("END");
  return out.join("\n") + "\n";
}
