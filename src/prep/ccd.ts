/**
 * The RCSB Chemical Component Dictionary, which is where a ligand's bonds come
 * from.
 *
 * 🔴 A PDB does not say what a ligand *is*. Its `HETATM` records carry an
 * element, a name and three coordinates; there are no bonds and no bond orders.
 * Everything downstream of a ligand — typing it, protonating it, charging it —
 * needs the chemistry, and the only authority for a three-letter component code
 * is the CCD.
 *
 * The Python route fetches the component's ideal SDF and transfers its bond
 * orders onto the fragment with RDKit's `AssignBondOrdersFromTemplate`, which
 * is a substructure match. This reads the component's **CIF** instead and
 * transfers by **atom name**, because the fragment and the component share
 * them: the names in a PDB `HETATM` record are the CCD's own `atom_id` values.
 * That is simpler, needs no graph matching, and — unlike a substructure match —
 * cannot pick the wrong symmetry-equivalent mapping. `test/ligand.test.ts`
 * checks the two routes agree bond for bond.
 *
 * Only the two loops that matter are parsed, `_chem_comp_atom` and
 * `_chem_comp_bond`. A CIF is a general format and this is not a general
 * reader: it handles the loop forms the CCD actually emits, and refuses
 * anything else rather than guessing.
 */

export type BondOrder = "SING" | "DOUB" | "TRIP" | "AROM";

export interface ComponentAtom {
  /** The CCD's `atom_id`, which is also the PDB atom name. */
  readonly name: string;
  readonly element: string;
  readonly formalCharge: number;
  readonly aromatic: boolean;
  /**
   * `pdbx_leaving_atom_flag`. A leaving atom is removed when the component is
   * linked into a polymer -- the OXT of a residue, for instance -- and is
   * absent from most deposited fragments. Kept so the matcher can tell "this
   * atom is missing because it left" from "this atom is missing".
   */
  readonly leaving: boolean;
}

export interface ComponentBond {
  readonly from: string;
  readonly to: string;
  readonly order: BondOrder;
  readonly aromatic: boolean;
}

export interface ChemicalComponent {
  readonly id: string;
  readonly name?: string;
  readonly formula?: string;
  readonly atoms: readonly ComponentAtom[];
  readonly bonds: readonly ComponentBond[];
}

export class CcdParseError extends Error {
  constructor(detail: string) {
    super(
      `this does not look like a CCD component definition: ${detail}. The file ` +
        "wanted is the one at " +
        "https://files.rcsb.org/ligands/download/<CODE>.cif",
    );
    this.name = "CcdParseError";
  }
}

/**
 * Split a CIF data line into values.
 *
 * Quoted values may contain spaces, and the CCD uses both `'` and `"`. `?` and
 * `.` are CIF's null markers and become the empty string, which the callers
 * then treat as absent -- a formal charge of `?` is not zero in principle, but
 * the CCD writes `0` when it means zero.
 */
function splitCifLine(line: string): string[] {
  const values: string[] = [];
  let at = 0;
  while (at < line.length) {
    while (at < line.length && /\s/.test(line[at] as string)) at++;
    if (at >= line.length) break;
    const quote = line[at];
    if (quote === "'" || quote === '"') {
      at++;
      let value = "";
      while (at < line.length && line[at] !== quote) {
        value += line[at];
        at++;
      }
      at++;
      values.push(value);
      continue;
    }
    let value = "";
    while (at < line.length && !/\s/.test(line[at] as string)) {
      value += line[at];
      at++;
    }
    values.push(value === "?" || value === "." ? "" : value);
  }
  return values;
}

/** Collect one `loop_` whose tags share `prefix`, as rows keyed by tag. */
function readLoop(
  lines: readonly string[],
  prefix: string,
): readonly Readonly<Record<string, string>>[] {
  const rows: Record<string, string>[] = [];
  for (let index = 0; index < lines.length; index++) {
    if (lines[index]?.trim() !== "loop_") continue;
    const tags: string[] = [];
    let at = index + 1;
    while (at < lines.length && (lines[at] as string).trim().startsWith("_")) {
      tags.push((lines[at] as string).trim().split(/\s+/)[0] as string);
      at++;
    }
    if (!tags.some((tag) => tag.startsWith(prefix))) continue;
    const columns = tags.map((tag) => tag.slice(tag.indexOf(".") + 1));
    for (; at < lines.length; at++) {
      const line = lines[at] as string;
      const trimmed = line.trim();
      if (trimmed === "" || trimmed === "#" || trimmed.startsWith("loop_")) break;
      if (trimmed.startsWith("_")) break;
      const values = splitCifLine(line);
      if (values.length !== columns.length) {
        throw new CcdParseError(
          `a row of ${prefix} has ${values.length} values for ` +
            `${columns.length} columns`,
        );
      }
      const row: Record<string, string> = {};
      for (const [slot, column] of columns.entries()) {
        row[column] = values[slot] as string;
      }
      rows.push(row);
    }
    return rows;
  }
  return rows;
}

function asOrder(value: string): BondOrder {
  if (value === "SING" || value === "DOUB" || value === "TRIP" || value === "AROM") {
    return value;
  }
  throw new CcdParseError(`bond order "${value}" is not one of SING/DOUB/TRIP/AROM`);
}

export function parseChemicalComponent(cif: string): ChemicalComponent {
  const lines = cif.split("\n");
  const data = lines.find((line) => line.startsWith("data_"));
  if (data === undefined) throw new CcdParseError("there is no data_ block");
  const id = data.slice("data_".length).trim();

  const atomRows = readLoop(lines, "_chem_comp_atom.");
  if (atomRows.length === 0) throw new CcdParseError("no _chem_comp_atom loop");
  const atoms: ComponentAtom[] = atomRows.map((row) => {
    const name = row["atom_id"];
    const element = row["type_symbol"];
    if (name === undefined || element === undefined) {
      throw new CcdParseError("_chem_comp_atom has no atom_id or type_symbol");
    }
    return {
      name,
      // The CCD writes elements upper-case (`CL`); everything here compares
      // case-insensitively, and this keeps the file's own spelling.
      element,
      formalCharge: Number(row["charge"] ?? "0") || 0,
      aromatic: row["pdbx_aromatic_flag"] === "Y",
      leaving: row["pdbx_leaving_atom_flag"] === "Y",
    };
  });

  const bondRows = readLoop(lines, "_chem_comp_bond.");
  const known = new Set(atoms.map((atom) => atom.name));
  const bonds: ComponentBond[] = bondRows.map((row) => {
    const from = row["atom_id_1"];
    const to = row["atom_id_2"];
    if (from === undefined || to === undefined) {
      throw new CcdParseError("_chem_comp_bond has no atom_id_1 or atom_id_2");
    }
    if (!known.has(from) || !known.has(to)) {
      throw new CcdParseError(
        `bond ${from}-${to} names an atom the component does not define`,
      );
    }
    return {
      from,
      to,
      order: asOrder(row["value_order"] ?? ""),
      aromatic: row["pdbx_aromatic_flag"] === "Y",
    };
  });

  // A component with no bond loop is a single atom -- a metal ion -- and that
  // is legitimate. More than one atom and no bonds is not.
  if (bonds.length === 0 && atoms.length > 1) {
    throw new CcdParseError(
      `${id} has ${atoms.length} atoms and no _chem_comp_bond loop`,
    );
  }
  return { id, atoms, bonds };
}

/** The URL a component's definition lives at. */
export const componentUrl = (code: string): string =>
  `https://files.rcsb.org/ligands/download/${code.toUpperCase()}.cif`;
