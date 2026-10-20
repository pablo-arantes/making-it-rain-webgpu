/**
 * The bonds a PDB implies.
 *
 * A PDB does not say what is bonded to what -- CONECT records are optional and
 * usually absent for standard residues -- so the topology is reconstructed from
 * a table of standard bonds per residue name. That table is OpenMM's own
 * `residues.xml`, vendored, because reconstructing it from chemistry would be a
 * second source of truth for something already settled.
 *
 * 🔴 The table's `-` and `+` prefixes are what make a polymer. `ALA` lists
 * `<Bond from="-C" to="N"/>`, meaning "the C of the *previous* residue in this
 * chain to my N". Without them every residue is a separate molecule: no peptide
 * bonds, no backbone torsions, and a template matcher that sees twenty free
 * amino acids rather than a chain. The prefix is only honoured inside a chain,
 * so a `TER` record genuinely ends the polymer.
 *
 * Disulfides are found by distance, as OpenMM does it, because no table can
 * know them: two cysteine sulfurs within 0.3 nm and neither carrying a hydrogen
 * are taken to be bonded.
 */

import { XmlScanner } from "../engine/xml.js";
import type { PdbTopology } from "../analysis/pdb.js";

/** `from`/`to` atom names, possibly prefixed with `-` or `+`. */
export type StandardBondTable = ReadonlyMap<
  string,
  readonly (readonly [string, string])[]
>;

export function parseStandardBonds(xml: string): StandardBondTable {
  const table = new Map<string, [string, string][]>();
  const scanner = new XmlScanner(xml);
  let current: string | null = null;
  while (scanner.next()) {
    if (scanner.closing) {
      if (scanner.name === "Residue") current = null;
      continue;
    }
    if (scanner.name === "Residue") {
      current = scanner.str("name");
      if (!table.has(current)) table.set(current, []);
      if (scanner.selfClosing) current = null;
    } else if (scanner.name === "Bond" && current !== null) {
      (table.get(current) as [string, string][]).push([
        scanner.str("from"),
        scanner.str("to"),
      ]);
    }
  }
  return table;
}

/** How far apart two cysteine sulfurs may be and still be called bonded, nm. */
export const DISULFIDE_DISTANCE = 0.3;

const SULFUR_NAMES = new Set(["SG"]);
const CYSTEINE_NAMES = new Set(["CYS", "CYX", "CYM"]);

/**
 * Every bond in a PDB: standard, disulfide, and whatever CONECT asserted.
 *
 * Returned as a sorted list of index pairs with the lower index first, so two
 * correct implementations produce the same list regardless of the order they
 * found things in.
 */
export function inferBonds(
  topology: PdbTopology,
  table: StandardBondTable,
): readonly (readonly [number, number])[] {
  const bonds = new Set<string>();
  const add = (a: number, b: number): void => {
    if (a === b) return;
    bonds.add(a < b ? `${a},${b}` : `${b},${a}`);
  };

  // --- Standard bonds, chain by chain ------------------------------------
  for (const chain of topology.chains) {
    // An atom-name index per residue, so a `-`/`+` prefix can reach sideways.
    const byName = chain.residues.map((residueIndex) => {
      const map = new Map<string, number>();
      for (const atom of (topology.residues[residueIndex] as { atoms: readonly number[] })
        .atoms) {
        map.set((topology.atoms[atom] as { name: string }).name, atom);
      }
      return map;
    });

    for (const [position, residueIndex] of chain.residues.entries()) {
      const residue = topology.residues[residueIndex] as { name: string };
      const entries = table.get(residue.name);
      if (entries === undefined) continue;
      for (const [rawFrom, rawTo] of entries) {
        const resolve = (name: string): number | undefined => {
          if (name.startsWith("-")) {
            if (position === 0) return undefined;
            return (byName[position - 1] as Map<string, number>).get(name.slice(1));
          }
          if (name.startsWith("+")) {
            if (position === chain.residues.length - 1) return undefined;
            return (byName[position + 1] as Map<string, number>).get(name.slice(1));
          }
          return (byName[position] as Map<string, number>).get(name);
        };
        const first = resolve(rawFrom);
        const second = resolve(rawTo);
        // A bond whose atoms are not both present is simply absent -- a
        // terminal residue has no next C, and a structure missing a side-chain
        // atom has no bond to it.
        if (first !== undefined && second !== undefined) add(first, second);
      }
    }
  }

  // --- Disulfides, by distance -------------------------------------------
  const sulfurs: number[] = [];
  for (const [residueIndex, residue] of topology.residues.entries()) {
    if (!CYSTEINE_NAMES.has(residue.name)) continue;
    for (const atom of residue.atoms) {
      const entry = topology.atoms[atom] as { name: string };
      if (!SULFUR_NAMES.has(entry.name)) continue;
      // A sulfur already carrying a hydrogen is a free thiol, not half a
      // disulfide, and OpenMM uses the same test.
      const hasHydrogen = [...bonds].some((key) => {
        const [a, b] = key.split(",").map(Number) as [number, number];
        const other = a === atom ? b : b === atom ? a : -1;
        return other >= 0 && (topology.atoms[other] as { element: string }).element === "H";
      });
      if (!hasHydrogen) sulfurs.push(atom);
      void residueIndex;
    }
  }
  for (let i = 0; i < sulfurs.length; i++) {
    for (let j = i + 1; j < sulfurs.length; j++) {
      const a = sulfurs[i] as number;
      const b = sulfurs[j] as number;
      let squared = 0;
      for (let axis = 0; axis < 3; axis++) {
        squared +=
          ((topology.positions[a * 3 + axis] as number) -
            (topology.positions[b * 3 + axis] as number)) **
          2;
      }
      if (Math.sqrt(squared) < DISULFIDE_DISTANCE) add(a, b);
    }
  }

  // --- CONECT, for anything the table does not know ----------------------
  // Added only when not already present, which is OpenMM's rule: a CONECT
  // repeating a standard bond must not create a duplicate.
  for (const [a, b] of topology.conect) add(a, b);

  return [...bonds]
    .map((key) => key.split(",").map(Number) as [number, number])
    .sort((x, y) => x[0] - y[0] || x[1] - y[1]);
}
