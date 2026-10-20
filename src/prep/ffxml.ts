/**
 * OpenMM force field XML, parsed.
 *
 * An `ffxml` is the whole of a force field: the atom types, a template for
 * every residue, and a table per force. Assigning parameters from it is where
 * Phase 3's parity is won or lost, and most of the difficulty is in the
 * *matching* rules rather than in the file format.
 *
 * Four of those rules, because a first reading gets each of them wrong:
 *
 * **A type has both a name and a class**, and a parameter may be keyed by
 * either. `<Bond type1="protein-C" .../>` matches that one type;
 * `<Bond class1="C" .../>` matches every type whose class is `C`. A table
 * usually mixes the two, and a class entry is the more general of the pair.
 *
 * **An empty string is a wildcard, and only in torsions.** `type1=""` matches
 * any atom. It is not "no atom" and it is not an omitted attribute -- the file
 * writes the attribute with an empty value, so a parser that treats missing and
 * empty alike will match torsions it should not.
 *
 * **Charges come from the residue, not the type.**
 * `<UseAttributeFromResidue name="charge"/>` in `<NonbondedForce>` says that
 * `sigma` and `epsilon` are per type while `charge` is per *atom of a template*.
 * Two alanines' CB atoms share a type and a charge here, but in a force field
 * with charge variation they would not, and the indirection is what allows it.
 *
 * **`ordering="amber"` is a real difference.** When several torsion entries
 * match a dihedral, Amber's rule is not "the first" or "the most specific" --
 * see `src/prep/torsions.ts`, which implements it against OpenMM's source.
 *
 * This parses; it does not match. Matching is the next file, so that a parsing
 * bug and a matching bug cannot be confused for one another.
 */

import { XmlScanner } from "../engine/xml.js";

export interface AtomType {
  readonly name: string;
  readonly className: string;
  readonly element: string | null;
  /** amu. */
  readonly mass: number;
}

export interface TemplateAtom {
  readonly name: string;
  readonly type: string;
  /** e, when the template carries one. */
  readonly charge: number | null;
}

export interface ResidueTemplate {
  readonly name: string;
  readonly atoms: readonly TemplateAtom[];
  /** Pairs of indices into `atoms`. */
  readonly bonds: readonly (readonly [number, number])[];
  /** Indices of atoms that bond to the next or previous residue. */
  readonly externalBonds: readonly number[];
  /** Whether the file marked this a variant to be matched only when asked. */
  readonly override: number;
}

/**
 * One row of a force table.
 *
 * `keys` holds the type names or class names in order, with `null` for a
 * wildcard, and `byClass` says which of the two a slot was written as -- the
 * distinction a matcher needs and a parser must not collapse.
 */
export interface ParameterRow {
  readonly keys: readonly (string | null)[];
  readonly byClass: readonly boolean[];
  readonly values: Readonly<Record<string, number>>;
}

export interface TorsionRow extends ParameterRow {
  /** k, periodicity and phase, one triple per term. */
  readonly terms: readonly {
    readonly k: number;
    readonly periodicity: number;
    readonly phase: number;
  }[];
  /**
   * `amber`, `charmm`, `smirnoff` or `default`, from the enclosing
   * `<PeriodicTorsionForce ordering=...>`.
   *
   * Carried per row rather than per force field, which is how OpenMM does it:
   * `registerImproperTorsion` stamps the section's ordering onto each row. It
   * matters as soon as two files are merged -- ff14SB declares `amber` and
   * tip3p declares nothing -- because a single field-wide value would apply
   * one file's tie-breaking rules to the other file's impropers.
   */
  readonly ordering: string;
}

export interface ForceField {
  readonly types: ReadonlyMap<string, AtomType>;
  /** Type names grouped by class, for class-keyed parameters. */
  readonly typesByClass: ReadonlyMap<string, readonly string[]>;
  readonly templates: readonly ResidueTemplate[];
  readonly bonds: readonly ParameterRow[];
  readonly angles: readonly ParameterRow[];
  readonly propers: readonly TorsionRow[];
  readonly impropers: readonly TorsionRow[];
  /** sigma and epsilon per type name. */
  readonly nonbonded: ReadonlyMap<string, { readonly sigma: number; readonly epsilon: number }>;
  readonly coulomb14Scale: number;
  readonly lj14Scale: number;
  /** Attributes `<NonbondedForce>` takes from the residue template. */
  readonly fromResidue: readonly string[];
  /** `amber` or `default`, which changes how competing torsions are resolved. */
  readonly torsionOrdering: string;
}

export class UnsupportedForceField extends Error {
  constructor(what: string) {
    super(
      `this force field uses ${what}, which the browser preparation path does ` +
        "not implement. The supported set is what the notebook's protein and " +
        "water force fields need: atom types, residue templates, harmonic " +
        "bonds and angles, periodic torsions and a Lennard-Jones/Coulomb " +
        "nonbonded force. An unsupported term would be silently missing from " +
        "every energy, so it is refused instead. Upload a prepared " +
        "`system.xml` to run anything else.",
    );
    this.name = "UnsupportedForceField";
  }
}

/** Force tags that would change the energy and are not implemented. */
const REFUSED = new Set([
  "GBSAOBCForce",
  "CustomGBForce",
  "CustomNonbondedForce",
  "CustomBondForce",
  "CustomAngleForce",
  "CustomTorsionForce",
  "RBTorsionForce",
  "CMAPTorsionForce",
  "LennardJonesForce",
  "DrudeForce",
  "AmoebaMultipoleForce",
]);

/** Slots a torsion names, in order. */
const TORSION_SLOTS = [1, 2, 3, 4] as const;

function readKeys(
  scanner: XmlScanner,
  count: number,
): { keys: (string | null)[]; byClass: boolean[] } {
  const keys: (string | null)[] = [];
  const byClass: boolean[] = [];
  for (let slot = 1; slot <= count; slot++) {
    const type = scanner.attr(`type${slot}`);
    const className = scanner.attr(`class${slot}`);
    if (className !== undefined) {
      // A class entry generalises over every type in that class.
      byClass.push(true);
      keys.push(className === "" ? null : className);
    } else {
      byClass.push(false);
      // 🔴 Present-and-empty is a wildcard; absent is also treated as one,
      // because OpenMM's own reader does that for torsions. The distinction
      // that matters is empty-versus-named, not absent-versus-empty.
      keys.push(type === undefined || type === "" ? null : type);
    }
  }
  return { keys, byClass };
}

function readTerms(scanner: XmlScanner): TorsionRow["terms"] {
  const terms: { k: number; periodicity: number; phase: number }[] = [];
  // Terms are numbered from one and stop at the first gap, which is how OpenMM
  // writes them: `k1 periodicity1 phase1 k2 ...`.
  for (let index = 1; ; index++) {
    const k = scanner.attr(`k${index}`);
    if (k === undefined) break;
    terms.push({
      k: Number(k),
      periodicity: scanner.num(`periodicity${index}`),
      phase: scanner.num(`phase${index}`),
    });
  }
  return terms;
}

/**
 * Parse one ffxml.
 *
 * Several files are usually combined -- a protein force field plus a water
 * model -- which `mergeForceFields` does; this reads one.
 */
export function parseForceField(xml: string): ForceField {
  const scanner = new XmlScanner(xml);
  const types = new Map<string, AtomType>();
  const templates: ResidueTemplate[] = [];
  const bonds: ParameterRow[] = [];
  const angles: ParameterRow[] = [];
  const propers: TorsionRow[] = [];
  const impropers: TorsionRow[] = [];
  const nonbonded = new Map<string, { sigma: number; epsilon: number }>();
  const fromResidue: string[] = [];
  let coulomb14Scale = 0.8333333333333334;
  let lj14Scale = 0.5;
  let torsionOrdering = "default";

  // Which container the scanner is inside, since the element names repeat --
  // `<Atom>` means a template atom inside `<Residues>` and a Lennard-Jones
  // entry inside `<NonbondedForce>`, and `<Bond>` likewise.
  let section = "";
  let residue: {
    name: string;
    atoms: TemplateAtom[];
    bonds: [number, number][];
    externalBonds: number[];
    override: number;
  } | null = null;

  const finishResidue = (): void => {
    if (residue === null) return;
    templates.push({
      name: residue.name,
      atoms: residue.atoms,
      bonds: residue.bonds,
      externalBonds: residue.externalBonds,
      override: residue.override,
    });
    residue = null;
  };

  while (scanner.next()) {
    const name = scanner.name;
    if (scanner.closing) {
      if (name === "Residue") finishResidue();
      else if (name === section) section = "";
      continue;
    }
    if (REFUSED.has(name)) throw new UnsupportedForceField(`<${name}>`);

    switch (name) {
      case "AtomTypes":
      case "Residues":
      case "HarmonicBondForce":
      case "HarmonicAngleForce":
      case "PeriodicTorsionForce":
      case "NonbondedForce":
        section = name;
        if (name === "PeriodicTorsionForce") {
          torsionOrdering = scanner.attr("ordering") ?? "default";
        }
        if (name === "NonbondedForce") {
          coulomb14Scale = scanner.numOr("coulomb14scale", coulomb14Scale);
          lj14Scale = scanner.numOr("lj14scale", lj14Scale);
        }
        break;

      case "Type": {
        const typeName = scanner.str("name");
        types.set(typeName, {
          name: typeName,
          className: scanner.attr("class") ?? typeName,
          element: scanner.attr("element") ?? null,
          mass: scanner.num("mass"),
        });
        break;
      }

      case "Residue":
        finishResidue();
        residue = {
          name: scanner.str("name"),
          atoms: [],
          bonds: [],
          externalBonds: [],
          override: Math.trunc(scanner.numOr("override", 0)),
        };
        break;

      case "Atom":
        if (section === "Residues" && residue !== null) {
          const charge = scanner.attr("charge");
          residue.atoms.push({
            name: scanner.str("name"),
            type: scanner.str("type"),
            charge: charge === undefined ? null : Number(charge),
          });
        } else if (section === "NonbondedForce") {
          const typeName = scanner.attr("type");
          const className = scanner.attr("class");
          const entry = {
            sigma: scanner.num("sigma"),
            epsilon: scanner.num("epsilon"),
          };
          if (typeName !== undefined) nonbonded.set(typeName, entry);
          else if (className !== undefined) {
            // Resolved after the types are known; stashed under a marker the
            // merge step expands.
            nonbonded.set(`class:${className}`, entry);
          }
        }
        break;

      case "Bond":
        if (section === "Residues" && residue !== null) {
          const byName = scanner.attr("atomName1");
          if (byName !== undefined) {
            const first = residue.atoms.findIndex((a) => a.name === byName);
            const second = residue.atoms.findIndex(
              (a) => a.name === scanner.str("atomName2"),
            );
            residue.bonds.push([first, second]);
          } else {
            // Older files index the template's atom list directly.
            residue.bonds.push([scanner.int("from"), scanner.int("to")]);
          }
        } else if (section === "HarmonicBondForce") {
          const { keys, byClass } = readKeys(scanner, 2);
          bonds.push({
            keys,
            byClass,
            values: { length: scanner.num("length"), k: scanner.num("k") },
          });
        }
        break;

      case "ExternalBond":
        if (residue !== null) {
          const byName = scanner.attr("atomName");
          residue.externalBonds.push(
            byName === undefined
              ? scanner.int("from")
              : residue.atoms.findIndex((a) => a.name === byName),
          );
        }
        break;

      case "Angle": {
        const { keys, byClass } = readKeys(scanner, 3);
        angles.push({
          keys,
          byClass,
          values: { angle: scanner.num("angle"), k: scanner.num("k") },
        });
        break;
      }

      case "Proper":
      case "Improper": {
        const { keys, byClass } = readKeys(scanner, TORSION_SLOTS.length);
        const row: TorsionRow = {
          keys,
          byClass,
          values: {},
          terms: readTerms(scanner),
          ordering: torsionOrdering,
        };
        if (name === "Proper") propers.push(row);
        else impropers.push(row);
        break;
      }

      case "UseAttributeFromResidue":
        fromResidue.push(scanner.str("name"));
        break;

      default:
        break;
    }
  }
  finishResidue();

  const typesByClass = new Map<string, string[]>();
  for (const type of types.values()) {
    const existing = typesByClass.get(type.className);
    if (existing === undefined) typesByClass.set(type.className, [type.name]);
    else existing.push(type.name);
  }

  // Expand the class-keyed Lennard-Jones entries now that the classes are known.
  for (const [key, entry] of [...nonbonded]) {
    if (!key.startsWith("class:")) continue;
    nonbonded.delete(key);
    for (const typeName of typesByClass.get(key.slice(6)) ?? []) {
      if (!nonbonded.has(typeName)) nonbonded.set(typeName, entry);
    }
  }

  return {
    types,
    typesByClass,
    templates,
    bonds,
    angles,
    propers,
    impropers,
    nonbonded,
    coulomb14Scale,
    lj14Scale,
    fromResidue,
    torsionOrdering,
  };
}

/**
 * Combine several files, as `ForceField(a, b)` does.
 *
 * Later files win on a clash, which is how a water model adds its residues
 * without disturbing the protein's. OpenMM's own rule is more subtle for
 * templates -- `override` picks between variants of the same residue name --
 * and that attribute is carried through for the matcher to use.
 */
export function mergeForceFields(fields: readonly ForceField[]): ForceField {
  if (fields.length === 0) throw new Error("no force field files were given");
  const types = new Map<string, AtomType>();
  const nonbonded = new Map<string, { sigma: number; epsilon: number }>();
  const templates: ResidueTemplate[] = [];
  const bonds: ParameterRow[] = [];
  const angles: ParameterRow[] = [];
  const propers: TorsionRow[] = [];
  const impropers: TorsionRow[] = [];
  const fromResidue = new Set<string>();
  let coulomb14Scale = (fields[0] as ForceField).coulomb14Scale;
  let lj14Scale = (fields[0] as ForceField).lj14Scale;
  let torsionOrdering = (fields[0] as ForceField).torsionOrdering;

  for (const field of fields) {
    for (const [name, type] of field.types) types.set(name, type);
    for (const [name, entry] of field.nonbonded) nonbonded.set(name, entry);
    templates.push(...field.templates);
    bonds.push(...field.bonds);
    angles.push(...field.angles);
    propers.push(...field.propers);
    impropers.push(...field.impropers);
    for (const attribute of field.fromResidue) fromResidue.add(attribute);
    coulomb14Scale = field.coulomb14Scale;
    lj14Scale = field.lj14Scale;
    if (field.torsionOrdering !== "default") torsionOrdering = field.torsionOrdering;
  }

  const typesByClass = new Map<string, string[]>();
  for (const type of types.values()) {
    const existing = typesByClass.get(type.className);
    if (existing === undefined) typesByClass.set(type.className, [type.name]);
    else existing.push(type.name);
  }

  return {
    types,
    typesByClass,
    templates,
    bonds,
    angles,
    propers,
    impropers,
    nonbonded,
    coulomb14Scale,
    lj14Scale,
    fromResidue: [...fromResidue],
    torsionOrdering,
  };
}
