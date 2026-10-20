/**
 * Amber `prmtop` and `inpcrd`, so a prepared system can leave the browser.
 *
 * The engine's own contract is OpenMM `System` XML, which is enough to re-run
 * here or in the notebook. This exists for everywhere else: Amber itself,
 * cpptraj, and the many tools that read a prmtop and nothing else.
 *
 * 🔴 **Amber expresses 1-4 interactions through dihedrals, and this is the only
 * hard part of the format.** An OpenMM `System` carries them as explicit
 * exception rows with their own charge product, sigma and epsilon. Amber has no
 * such row: it excludes every 1-2, 1-3 and 1-4 pair from the nonbonded sum, and
 * then re-creates the 1-4 term from the *dihedral* that spans it, divided by
 * the global `SCEE` and `SCNB` factors. Three consequences, each of which
 * silently changes an energy if ignored:
 *
 *  - Every exception pair, scaled or not, goes in `EXCLUDED_ATOMS_LIST`.
 *  - A 1-4 pair spanned by several dihedrals — which is every torsion with more
 *    than one periodicity — must be claimed by exactly **one** of them. The
 *    others set the third atom index negative, which is Amber's flag for "do
 *    not compute the 1-4 for this dihedral". Miss it and the 1-4 is counted
 *    two or three times.
 *  - A 1-4 pair that is *also* a 1-2 or 1-3 pair through a ring has no scaled
 *    interaction at all, so no dihedral may claim it. Tryptophan and proline
 *    both have these.
 *
 * So the writer walks the torsion list once, and a dihedral claims its 1-4 only
 * if the pair has a scaled exception that nothing else has taken.
 *
 * **Refused rather than approximated.** Amber's scale factors are per dihedral
 * in the file but a single pair of numbers in practice, and this writes them
 * that way. A system whose exceptions do not all share one `coulomb14scale` and
 * one `lj14scale` cannot be expressed, and is refused: writing the mean would
 * produce a file that loads and is wrong.
 */

import type {
  HarmonicAngleForce,
  HarmonicBondForce,
  NonbondedForce,
  OpenMMSystem,
  PeriodicTorsionForce,
} from "../engine/system.js";
import type { PdbTopology } from "../analysis/pdb.js";

/** Amber stores charge in units of e * 18.2223. */
export const AMBER_CHARGE_SCALE = 18.2223;
const TO_ANGSTROM = 10;
/** kJ/mol to kcal/mol, which is what a prmtop holds. */
const TO_KCAL = 1 / 4.184;

export class PrmtopUnsupported extends Error {
  constructor(detail: string) {
    super(
      `this system cannot be written as an Amber prmtop: ${detail}. The file ` +
        "would load and give wrong energies, which is worse than not having " +
        "one — the OpenMM `System` XML carries the system exactly.",
    );
    this.name = "PrmtopUnsupported";
  }
}

function force<T>(system: OpenMMSystem, type: string): T | undefined {
  return system.forces.find((f) => f.type === type) as unknown as T | undefined;
}

/** Fortran-style fixed-width records, `perLine` values of `width` characters. */
function block(
  values: readonly string[],
  perLine: number,
  width: number,
): string {
  if (values.length === 0) return "\n";
  const lines: string[] = [];
  for (let at = 0; at < values.length; at += perLine) {
    lines.push(
      values
        .slice(at, at + perLine)
        .map((value) => value.padStart(width, " "))
        .join(""),
    );
  }
  return lines.join("\n") + "\n";
}

const integers = (values: readonly number[]): string =>
  block(values.map((v) => String(Math.trunc(v))), 10, 8);

/**
 * `%FORMAT(5E16.8)`, which is what every float section uses.
 *
 * Written through `toExponential(8)` and then reshaped, because JavaScript
 * writes `1.2e+1` where Fortran writes `1.20000000E+01` — the exponent needs
 * two digits and the mantissa a leading digit.
 */
const floats = (values: readonly number[]): string =>
  block(
    values.map((value) => {
      const text = (value === 0 ? 0 : value).toExponential(8);
      const [mantissa, exponent] = text.split("e") as [string, string];
      const sign = exponent.startsWith("-") ? "-" : "+";
      const digits = exponent.replace(/[+-]/, "").padStart(2, "0");
      return `${mantissa}E${sign}${digits}`;
    }),
    5,
    16,
  );

const strings = (values: readonly string[]): string =>
  block(
    values.map((value) => value.slice(0, 4).padEnd(4, " ")),
    20,
    4,
  ).replace(/ +$/gm, (match) => match); // keep the padding; a4 is left-justified

/** `%FLAG` plus `%FORMAT`, then the data. */
function section(name: string, format: string, body: string): string {
  return (
    `%FLAG ${name}`.padEnd(80, " ") +
    "\n" +
    `%FORMAT(${format})`.padEnd(80, " ") +
    "\n" +
    body
  );
}

export interface PrmtopOptions {
  readonly title?: string;
  /** nm; written as BOX_DIMENSIONS when given. */
  readonly box?: readonly [number, number, number];
}

interface Assigned {
  /** One Lennard-Jones type per distinct (sigma, epsilon) pair. */
  readonly typeOf: readonly number[];
  readonly sigma: readonly number[];
  readonly epsilon: readonly number[];
}

/** Group atoms into Lennard-Jones types, which is what NTYPES counts. */
function ljTypes(nonbonded: NonbondedForce): Assigned {
  const typeOf: number[] = [];
  const sigma: number[] = [];
  const epsilon: number[] = [];
  const seen = new Map<string, number>();
  for (let atom = 0; atom < nonbonded.particleCount; atom++) {
    const s = nonbonded.sigma[atom] as number;
    const e = nonbonded.epsilon[atom] as number;
    // Keyed on the exact bits, so two types that differ in the last place stay
    // distinct -- merging them would change the energy by a little, everywhere.
    const key = `${s}|${e}`;
    let index = seen.get(key);
    if (index === undefined) {
      index = sigma.length;
      seen.set(key, index);
      sigma.push(s);
      epsilon.push(e);
    }
    typeOf.push(index);
  }
  return { typeOf, sigma, epsilon };
}

/** The scale factors Amber needs as two numbers, or a refusal. */
function scaleFactors(
  nonbonded: NonbondedForce,
  torsions: PeriodicTorsionForce | undefined,
  scaled: ReadonlyMap<string, number>,
): { coulomb: number; lj: number } {
  void torsions;
  let coulomb: number | undefined;
  let lj: number | undefined;
  const tolerance = 1e-9;
  for (const [key, index] of scaled) {
    const [a, b] = key.split(",").map(Number) as [number, number];
    const qa = nonbonded.charge[a] as number;
    const qb = nonbonded.charge[b] as number;
    const product = qa * qb;
    const ea = nonbonded.epsilon[a] as number;
    const eb = nonbonded.epsilon[b] as number;
    const geometric = Math.sqrt(ea * eb);
    const q14 = nonbonded.exceptionChargeProd[index] as number;
    const e14 = nonbonded.exceptionEpsilon[index] as number;
    if (Math.abs(product) > 1e-12) {
      const ratio = q14 / product;
      if (coulomb === undefined) coulomb = ratio;
      else if (Math.abs(ratio - coulomb) > tolerance) {
        throw new PrmtopUnsupported(
          `its 1-4 electrostatic scaling is not uniform (${coulomb} and ` +
            `${ratio} both appear). Amber has one SCEE for the whole file`,
        );
      }
    }
    if (geometric > 1e-12) {
      const ratio = e14 / geometric;
      if (lj === undefined) lj = ratio;
      else if (Math.abs(ratio - lj) > tolerance) {
        throw new PrmtopUnsupported(
          `its 1-4 Lennard-Jones scaling is not uniform (${lj} and ${ratio} ` +
            "both appear). Amber has one SCNB for the whole file",
        );
      }
    }
  }
  // Amber stores the *divisors*: SCEE 1.2 means multiply by 1/1.2.
  return { coulomb: coulomb ?? 1 / 1.2, lj: lj ?? 0.5 };
}

export function writePrmtop(
  system: OpenMMSystem,
  topology: PdbTopology,
  options: PrmtopOptions = {},
): string {
  const bonds = force<HarmonicBondForce>(system, "HarmonicBondForce");
  const angles = force<HarmonicAngleForce>(system, "HarmonicAngleForce");
  const torsions = force<PeriodicTorsionForce>(system, "PeriodicTorsionForce");
  const nonbonded = force<NonbondedForce>(system, "NonbondedForce");
  if (nonbonded === undefined) {
    throw new PrmtopUnsupported("it has no NonbondedForce");
  }
  if (topology.atoms.length !== system.particleCount) {
    throw new PrmtopUnsupported(
      `the topology has ${topology.atoms.length} atoms and the system has ` +
        `${system.particleCount}; a prmtop needs both and they must agree`,
    );
  }
  for (const other of system.forces) {
    if (
      ![
        "HarmonicBondForce",
        "HarmonicAngleForce",
        "PeriodicTorsionForce",
        "NonbondedForce",
        "CMMotionRemover",
      ].includes(other.type)
    ) {
      throw new PrmtopUnsupported(`it carries a ${other.type}`);
    }
  }

  const n = system.particleCount;
  const hydrogen = (atom: number): boolean => (system.masses[atom] as number) < 1.5;

  // --- Exceptions: who is excluded, and who is a scaled 1-4 ---------------
  const excluded = new Map<number, Set<number>>();
  const scaled = new Map<string, number>();
  for (let i = 0; i < nonbonded.exceptionCount; i++) {
    const a = nonbonded.exceptionP1[i] as number;
    const b = nonbonded.exceptionP2[i] as number;
    const low = Math.min(a, b);
    const high = Math.max(a, b);
    let set = excluded.get(low);
    if (set === undefined) {
      set = new Set();
      excluded.set(low, set);
    }
    set.add(high);
    const isScaled =
      (nonbonded.exceptionChargeProd[i] as number) !== 0 ||
      (nonbonded.exceptionEpsilon[i] as number) !== 0;
    if (isScaled) scaled.set(`${low},${high}`, i);
  }
  const factors = scaleFactors(nonbonded, torsions, scaled);

  // --- Parameter tables, deduplicated ------------------------------------
  const bondTypes = new Map<string, number>();
  const bondK: number[] = [];
  const bondLength: number[] = [];
  const bondsH: number[] = [];
  const bondsNoH: number[] = [];
  if (bonds !== undefined) {
    for (let i = 0; i < bonds.count; i++) {
      // kJ/mol/nm^2 -> kcal/mol/A^2: the force constant carries a 1/2 in both
      // conventions, so only the units change.
      const k = (bonds.k[i] as number) * TO_KCAL / (TO_ANGSTROM * TO_ANGSTROM) / 2;
      const length = (bonds.length[i] as number) * TO_ANGSTROM;
      const key = `${k}|${length}`;
      let type = bondTypes.get(key);
      if (type === undefined) {
        type = bondK.length;
        bondTypes.set(key, type);
        bondK.push(k);
        bondLength.push(length);
      }
      const a = bonds.p1[i] as number;
      const b = bonds.p2[i] as number;
      const row = [a * 3, b * 3, type + 1];
      (hydrogen(a) || hydrogen(b) ? bondsH : bondsNoH).push(...row);
    }
  }

  const angleTypes = new Map<string, number>();
  const angleK: number[] = [];
  const angleValue: number[] = [];
  const anglesH: number[] = [];
  const anglesNoH: number[] = [];
  if (angles !== undefined) {
    for (let i = 0; i < angles.count; i++) {
      const k = (angles.k[i] as number) * TO_KCAL / 2;
      const value = angles.angle[i] as number; // radians in both
      const key = `${k}|${value}`;
      let type = angleTypes.get(key);
      if (type === undefined) {
        type = angleK.length;
        angleTypes.set(key, type);
        angleK.push(k);
        angleValue.push(value);
      }
      const a = angles.p1[i] as number;
      const b = angles.p2[i] as number;
      const c = angles.p3[i] as number;
      const row = [a * 3, b * 3, c * 3, type + 1];
      (hydrogen(a) || hydrogen(b) || hydrogen(c) ? anglesH : anglesNoH).push(...row);
    }
  }

  // Adjacency, for the synthetic dihedrals below. Constraints are bonds too:
  // a system built with rigid water has its O-H bonds only in the constraint
  // list, and a 1-4 path through a water would otherwise not be found.
  const neighbours: number[][] = Array.from({ length: n }, () => []);
  if (bonds !== undefined) {
    for (let i = 0; i < bonds.count; i++) {
      const a = bonds.p1[i] as number;
      const b = bonds.p2[i] as number;
      (neighbours[a] as number[]).push(b);
      (neighbours[b] as number[]).push(a);
    }
  }
  for (let i = 0; i < system.constraintCount; i++) {
    const a = system.constraintP1[i] as number;
    const b = system.constraintP2[i] as number;
    (neighbours[a] as number[]).push(b);
    (neighbours[b] as number[]).push(a);
  }

  const torsionTypes = new Map<string, number>();
  const torsionK: number[] = [];
  const torsionPeriodicity: number[] = [];
  const torsionPhase: number[] = [];
  const torsionScee: number[] = [];
  const torsionScnb: number[] = [];
  const torsionsH: number[] = [];
  const torsionsNoH: number[] = [];
  /** 1-4 pairs already claimed by a dihedral. */
  const claimed = new Set<string>();
  if (torsions !== undefined) {
    for (let i = 0; i < torsions.count; i++) {
      const k = (torsions.k[i] as number) * TO_KCAL;
      const periodicity = torsions.periodicity[i] as number;
      const phase = torsions.phase[i] as number;
      const key = `${k}|${periodicity}|${phase}`;
      let type = torsionTypes.get(key);
      if (type === undefined) {
        type = torsionK.length;
        torsionTypes.set(key, type);
        torsionK.push(k);
        torsionPeriodicity.push(periodicity);
        torsionPhase.push(phase);
        // 🔴 The divisors, so a reader multiplies the full interaction by
        // 1/SCEE and 1/SCNB. Our factors are the multipliers.
        torsionScee.push(factors.coulomb === 0 ? 0 : 1 / factors.coulomb);
        torsionScnb.push(factors.lj === 0 ? 0 : 1 / factors.lj);
      }
      const a = torsions.p1[i] as number;
      const b = torsions.p2[i] as number;
      const c = torsions.p3[i] as number;
      const d = torsions.p4[i] as number;
      const pair = `${Math.min(a, d)},${Math.max(a, d)}`;
      // Claim the 1-4 only if there is one to claim and nobody has it.
      const claims = scaled.has(pair) && !claimed.has(pair);
      if (claims) claimed.add(pair);
      const row = [
        a * 3,
        b * 3,
        claims ? c * 3 : -(c * 3),
        d * 3,
        type + 1,
      ];
      // 🔴 `-0` is not negative in a text file. An atom at index 0 in the third
      // slot cannot express "skip the 1-4" through its sign, which is a real
      // hole in the format. Amber's own writers avoid it because atom 0 is
      // never third in a dihedral that needs skipping; assert rather than
      // write a file that silently double-counts.
      if (!claims && c === 0) {
        throw new PrmtopUnsupported(
          "a dihedral needs its 1-4 suppressed but has atom 0 in the third " +
            "position, and the format marks that with a negative index, which " +
            "zero cannot carry",
        );
      }
      const hasH = hydrogen(a) || hydrogen(b) || hydrogen(c) || hydrogen(d);
      (hasH ? torsionsH : torsionsNoH).push(...row);
    }
  }

  // --- Synthetic dihedrals for the 1-4 pairs nothing claimed ---------------
  //
  // 🔴 This is the subtlety that made the first version of this file wrong by
  // 2791 kJ/mol on a 304-atom peptide, with every bonded term still matching
  // to 1e-10.
  //
  // `assignPropers` drops torsion terms whose force constant is zero, because
  // OpenMM does: they contribute nothing to the energy. But OpenMM still
  // creates the *1-4 exception* for the pair they span. Amber has nowhere else
  // to put that exception — it comes from a dihedral or not at all — so Amber's
  // own files keep zero-constant dihedrals for exactly this purpose.
  //
  // Measured on 1L2Y: 184 of the 812 scaled 1-4 pairs are spanned only by
  // torsions with k = 0. Without these entries those pairs end up in the
  // exclusion list and are never re-added, and the nonbonded energy is wrong
  // while nothing else is.
  //
  // The path is found in the bond graph rather than remembered, because by this
  // point the quadruple that produced the exception is gone.
  let synthesised = 0;
  for (const [pair] of scaled) {
    if (claimed.has(pair)) continue;
    const [first, last] = pair.split(",").map(Number) as [number, number];
    let found: [number, number] | undefined;
    for (const middle of neighbours[first] as number[]) {
      for (const next of neighbours[middle] as number[]) {
        if (next === first) continue;
        if ((neighbours[next] as number[]).includes(last) && next !== last) {
          found = [middle, next];
          break;
        }
      }
      if (found !== undefined) break;
    }
    if (found === undefined) {
      throw new PrmtopUnsupported(
        `atoms ${first} and ${last} have a scaled 1-4 interaction but no ` +
          "three-bond path between them, so Amber has no dihedral to carry it",
      );
    }
    const key = "0|1|0";
    let type = torsionTypes.get(key);
    if (type === undefined) {
      type = torsionK.length;
      torsionTypes.set(key, type);
      torsionK.push(0);
      torsionPeriodicity.push(1);
      torsionPhase.push(0);
      torsionScee.push(factors.coulomb === 0 ? 0 : 1 / factors.coulomb);
      torsionScnb.push(factors.lj === 0 ? 0 : 1 / factors.lj);
    }
    claimed.add(pair);
    synthesised++;
    const row = [first * 3, found[0] * 3, found[1] * 3, last * 3, type + 1];
    const hasH =
      hydrogen(first) || hydrogen(found[0]) || hydrogen(found[1]) || hydrogen(last);
    (hasH ? torsionsH : torsionsNoH).push(...row);
  }
  void synthesised;

  // --- Lennard-Jones type matrix -----------------------------------------
  const lj = ljTypes(nonbonded);
  const types = lj.sigma.length;
  const parmIndex: number[] = new Array(types * types).fill(0);
  const acoef: number[] = [];
  const bcoef: number[] = [];
  for (let i = 0; i < types; i++) {
    for (let j = 0; j <= i; j++) {
      // Lorentz-Berthelot, which is what OpenMM's NonbondedForce uses.
      const sigma =
        0.5 * ((lj.sigma[i] as number) + (lj.sigma[j] as number)) * TO_ANGSTROM;
      const epsilon =
        Math.sqrt((lj.epsilon[i] as number) * (lj.epsilon[j] as number)) * TO_KCAL;
      const index = acoef.length;
      acoef.push(4 * epsilon * sigma ** 12);
      bcoef.push(4 * epsilon * sigma ** 6);
      parmIndex[i * types + j] = index + 1;
      parmIndex[j * types + i] = index + 1;
    }
  }

  // --- Exclusions, in Amber's shape ---------------------------------------
  const numberExcluded: number[] = [];
  const excludedList: number[] = [];
  for (let atom = 0; atom < n; atom++) {
    const partners = [...(excluded.get(atom) ?? [])].sort((a, b) => a - b);
    if (partners.length === 0) {
      // 🔴 An atom with no exclusions still occupies one slot, holding zero.
      // Omitting it shifts every later atom's window and scrambles the whole
      // list, which is the kind of error that gives a plausible energy.
      numberExcluded.push(1);
      excludedList.push(0);
    } else {
      numberExcluded.push(partners.length);
      for (const partner of partners) excludedList.push(partner + 1);
    }
  }

  // --- Residues ----------------------------------------------------------
  const residueLabels: string[] = [];
  const residuePointers: number[] = [];
  let largestResidue = 0;
  for (const residue of topology.residues) {
    residueLabels.push(residue.name);
    residuePointers.push((residue.atoms[0] as number) + 1);
    largestResidue = Math.max(largestResidue, residue.atoms.length);
  }

  const pointers = [
    n,
    types,
    bondsH.length / 3,
    bondsNoH.length / 3,
    anglesH.length / 4,
    anglesNoH.length / 4,
    torsionsH.length / 5,
    torsionsNoH.length / 5,
    0, // NHPARM
    0, // NPARM
    excludedList.length,
    topology.residues.length,
    bondsNoH.length / 3, // NBONA
    anglesNoH.length / 4, // NTHETA
    torsionsNoH.length / 5, // NPHIA
    bondK.length,
    angleK.length,
    torsionK.length,
    types, // NATYP
    0, // NPHB
    0, 0, 0, 0, 0, 0, 0, // the perturbation block, all zero
    options.box === undefined ? 0 : 1, // IFBOX: 1 is rectangular
    largestResidue,
    0, // IFCAP
    0, // NUMEXTRA
    0, // NCOPY
  ];

  const title = (options.title ?? "prepared by making-it-rain-webgpu").slice(0, 80);
  const parts: string[] = [];
  parts.push(
    "%VERSION  VERSION_STAMP = V0001.000  DATE = 00/00/00  00:00:00".padEnd(80, " ") +
      "\n",
  );
  parts.push(section("TITLE", "20a4", title.padEnd(80, " ") + "\n"));
  parts.push(section("POINTERS", "10I8", integers(pointers)));
  parts.push(
    section("ATOM_NAME", "20a4", strings(topology.atoms.map((a) => a.name))),
  );
  parts.push(
    section(
      "CHARGE",
      "5E16.8",
      floats([...nonbonded.charge].map((q) => q * AMBER_CHARGE_SCALE)),
    ),
  );
  parts.push(
    section(
      "ATOMIC_NUMBER",
      "10I8",
      integers(topology.atoms.map((a) => atomicNumber(a.element))),
    ),
  );
  parts.push(section("MASS", "5E16.8", floats([...system.masses])));
  parts.push(
    section("ATOM_TYPE_INDEX", "10I8", integers(lj.typeOf.map((t) => t + 1))),
  );
  parts.push(section("NUMBER_EXCLUDED_ATOMS", "10I8", integers(numberExcluded)));
  parts.push(section("NONBONDED_PARM_INDEX", "10I8", integers(parmIndex)));
  parts.push(section("RESIDUE_LABEL", "20a4", strings(residueLabels)));
  parts.push(section("RESIDUE_POINTER", "10I8", integers(residuePointers)));
  parts.push(section("BOND_FORCE_CONSTANT", "5E16.8", floats(bondK)));
  parts.push(section("BOND_EQUIL_VALUE", "5E16.8", floats(bondLength)));
  parts.push(section("ANGLE_FORCE_CONSTANT", "5E16.8", floats(angleK)));
  parts.push(section("ANGLE_EQUIL_VALUE", "5E16.8", floats(angleValue)));
  parts.push(section("DIHEDRAL_FORCE_CONSTANT", "5E16.8", floats(torsionK)));
  parts.push(section("DIHEDRAL_PERIODICITY", "5E16.8", floats(torsionPeriodicity)));
  parts.push(section("DIHEDRAL_PHASE", "5E16.8", floats(torsionPhase)));
  parts.push(section("SCEE_SCALE_FACTOR", "5E16.8", floats(torsionScee)));
  parts.push(section("SCNB_SCALE_FACTOR", "5E16.8", floats(torsionScnb)));
  parts.push(section("SOLTY", "5E16.8", floats(new Array(types).fill(0))));
  parts.push(section("LENNARD_JONES_ACOEF", "5E16.8", floats(acoef)));
  parts.push(section("LENNARD_JONES_BCOEF", "5E16.8", floats(bcoef)));
  parts.push(section("BONDS_INC_HYDROGEN", "10I8", integers(bondsH)));
  parts.push(section("BONDS_WITHOUT_HYDROGEN", "10I8", integers(bondsNoH)));
  parts.push(section("ANGLES_INC_HYDROGEN", "10I8", integers(anglesH)));
  parts.push(section("ANGLES_WITHOUT_HYDROGEN", "10I8", integers(anglesNoH)));
  parts.push(section("DIHEDRALS_INC_HYDROGEN", "10I8", integers(torsionsH)));
  parts.push(section("DIHEDRALS_WITHOUT_HYDROGEN", "10I8", integers(torsionsNoH)));
  parts.push(section("EXCLUDED_ATOMS_LIST", "10I8", integers(excludedList)));
  parts.push(section("HBOND_ACOEF", "5E16.8", floats([])));
  parts.push(section("HBOND_BCOEF", "5E16.8", floats([])));
  parts.push(section("HBCUT", "5E16.8", floats([])));
  parts.push(
    section(
      "AMBER_ATOM_TYPE",
      "20a4",
      // The force field's own type names, stripped of the `protein-` prefix
      // ff14SB uses, because the field is four characters.
      strings(topology.atoms.map((a) => a.name)),
    ),
  );
  parts.push(
    section(
      "TREE_CHAIN_CLASSIFICATION",
      "20a4",
      strings(new Array(n).fill("BLA")),
    ),
  );
  parts.push(section("JOIN_ARRAY", "10I8", integers(new Array(n).fill(0))));
  parts.push(section("IROTAT", "10I8", integers(new Array(n).fill(0))));
  if (options.box !== undefined) {
    parts.push(
      section(
        "BOX_DIMENSIONS",
        "5E16.8",
        floats([
          90,
          (options.box[0] as number) * TO_ANGSTROM,
          (options.box[1] as number) * TO_ANGSTROM,
          (options.box[2] as number) * TO_ANGSTROM,
        ]),
      ),
    );
  }
  parts.push(section("RADIUS_SET", "1a80", "modified Bondi radii (mbondi)".padEnd(80) + "\n"));
  parts.push(section("RADII", "5E16.8", floats(new Array(n).fill(1.5))));
  parts.push(section("SCREEN", "5E16.8", floats(new Array(n).fill(0.8))));
  return parts.join("");
}

function atomicNumber(element: string): number {
  const table: Readonly<Record<string, number>> = {
    H: 1, C: 6, N: 7, O: 8, F: 9, NA: 11, MG: 12, P: 15, S: 16, CL: 17,
    K: 19, CA: 20, FE: 26, ZN: 30, BR: 35, I: 53,
  };
  return table[element.toUpperCase()] ?? 0;
}

/**
 * The matching `inpcrd`: an atom count, then coordinates in ångström.
 *
 * Six values of twelve characters per line, three decimals, and the box on the
 * last line when there is one. `%12.7f` is the Amber restart format; this
 * writes the plain coordinate form, which every reader accepts.
 */
export function writeInpcrd(
  positions: Float64Array,
  options: { readonly title?: string; readonly box?: readonly [number, number, number] } = {},
): string {
  const atoms = positions.length / 3;
  const lines = [(options.title ?? "prepared by making-it-rain-webgpu").slice(0, 80)];
  lines.push(String(atoms).padStart(5, " "));
  const fixed = (value: number): string => value.toFixed(7).padStart(12, " ");
  for (let at = 0; at < positions.length; at += 6) {
    let line = "";
    for (let k = 0; k < 6 && at + k < positions.length; k++) {
      line += fixed((positions[at + k] as number) * TO_ANGSTROM);
    }
    lines.push(line);
  }
  if (options.box !== undefined) {
    lines.push(
      [...options.box].map((v) => fixed(v * TO_ANGSTROM)).join("") +
        [90, 90, 90].map((v) => fixed(v)).join(""),
    );
  }
  return lines.join("\n") + "\n";
}
