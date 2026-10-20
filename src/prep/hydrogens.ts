/**
 * Adding the hydrogens a crystal structure does not have.
 *
 * A PDB entry from the RCSB is almost always heavy atoms only, and a force
 * field needs every hydrogen. This transcribes `Modeller.addHydrogens`: pick a
 * protonation variant per residue, work out which hydrogens that variant wants,
 * and insert the missing ones.
 *
 * 🔴 **Positions are a deviation, not a parity target, and that is OpenMM's
 * doing rather than a shortcut here.** Each hydrogen OpenMM adds starts at its
 * parent plus a 0.1 nm offset built from `random.random()`, and the whole set
 * is then energy-minimised. So OpenMM does not reproduce its own hydrogen
 * positions between runs unless the interpreter's RNG happens to be seeded.
 * There is nothing there to match. We place each hydrogen deterministically,
 * away from the parent's existing neighbours, and leave it to the minimiser the
 * user runs anyway. `docs/DEVIATIONS.md` records it.
 *
 * What *is* a parity target, and is compared atom by atom: which variant each
 * residue takes, which hydrogens are added, what they are named, where they sit
 * in the atom order, and the bonds created.
 *
 * 🔴 **The documented HID/HIE rule is unreachable in OpenMM, and matching means
 * reproducing that.** `addHydrogens`' docstring says a neutral histidine picks
 * HID or HIE "based on which one forms a better hydrogen bond". The code
 * initialises `nd1IsBonded` and `ne2IsBonded`, then -- in the loop that is
 * supposed to test NE2 -- assigns `nd1IsBonded = True` again. `ne2IsBonded` is
 * never set, so `if ne2IsBonded and not nd1IsBonded` can never fire and the
 * automatic path always yields HID. The hydrogen-bond search is dead code. We
 * choose HID for the same case, because agreeing with OpenMM is the point, and
 * the alternative would be a silent disagreement on every neutral histidine.
 */

import type { PdbAtom, PdbResidue, PdbTopology } from "../analysis/pdb.js";
import { XmlScanner } from "../engine/xml.js";
import type { Pair } from "./bonded.js";

export interface HydrogenDef {
  readonly name: string;
  /** The heavy atom it attaches to, by name. */
  readonly parent: string;
  /** Above this pH the hydrogen is absent; `Infinity` when unconditional. */
  readonly maxPh: number;
  /** Variants this hydrogen belongs to, or `null` for all of them. */
  readonly variants: readonly string[] | null;
  /**
   * `N`, `C`, `-`, or a combination. `null` means "anywhere".
   *
   * `-` means *not* terminal, which is the opposite of what the letter
   * suggests and is how `OXT`'s absence mid-chain is expressed.
   */
  readonly terminal: string | null;
}

export interface ResidueHydrogens {
  readonly name: string;
  readonly variants: readonly string[];
  readonly hydrogens: readonly HydrogenDef[];
}

export type HydrogenDefinitions = ReadonlyMap<string, ResidueHydrogens>;

export function parseHydrogenDefinitions(xml: string): HydrogenDefinitions {
  const scanner = new XmlScanner(xml);
  const definitions = new Map<string, ResidueHydrogens>();
  let current: { name: string; variants: string[]; hydrogens: HydrogenDef[] } | null =
    null;

  while (scanner.next()) {
    if (scanner.closing) {
      if (scanner.name === "Residue" && current !== null) {
        definitions.set(current.name, current);
        current = null;
      }
      continue;
    }
    switch (scanner.name) {
      case "Residue":
        current = { name: scanner.str("name"), variants: [], hydrogens: [] };
        break;
      case "Variant":
        current?.variants.push(scanner.str("name"));
        break;
      case "H": {
        if (current === null) break;
        const maxPh = scanner.attr("maxph");
        const variants = scanner.attr("variant");
        current.hydrogens.push({
          name: scanner.str("name"),
          parent: scanner.str("parent"),
          maxPh: maxPh === undefined ? Infinity : Number(maxPh),
          variants: variants === undefined ? null : variants.split(","),
          terminal: scanner.attr("terminal") ?? null,
        });
        break;
      }
      default:
        break;
    }
  }
  // `<Residue>` may be self-closing in principle; flush anything outstanding.
  if (current !== null) definitions.set(current.name, current);
  return definitions;
}

function isHydrogen(atom: PdbAtom): boolean {
  return atom.element.toUpperCase() === "H";
}

/**
 * Drop atoms, keeping the topology consistent.
 *
 * Needed before adding hydrogens to a file that already has some: an NMR
 * structure's hydrogen names may not be the force field's, and `addHydrogens`
 * never renames one it finds. OpenMM's fixture path does the same two things --
 * `deleteWater()` then delete every hydrogen -- so rebuilding from heavy atoms
 * is the comparable starting point.
 *
 * Takes a predicate rather than a list of names so that "drop the hydrogens"
 * and "drop the water" are the same function called twice, instead of two
 * functions that can disagree about how to renumber.
 */
export function stripAtoms(
  topology: PdbTopology,
  drop: (atom: PdbAtom) => boolean,
): { topology: PdbTopology; keptIndices: readonly number[] } {
  const remap = new Map<number, number>();
  const atoms: PdbAtom[] = [];
  const kept: number[] = [];
  const positions: number[] = [];
  for (const [index, atom] of topology.atoms.entries()) {
    if (drop(atom)) continue;
    remap.set(index, atoms.length);
    kept.push(index);
    atoms.push(atom);
    positions.push(
      topology.positions[index * 3] as number,
      topology.positions[index * 3 + 1] as number,
      topology.positions[index * 3 + 2] as number,
    );
  }

  // Residues and chains that lose every atom disappear, which keeps "first
  // residue of the chain" -- the N-terminal test -- meaning what it says.
  const residueRemap = new Map<number, number>();
  const residues: PdbResidue[] = [];
  for (const [index, residue] of topology.residues.entries()) {
    const remaining = residue.atoms
      .map((a) => remap.get(a))
      .filter((a): a is number => a !== undefined);
    if (remaining.length === 0) continue;
    residueRemap.set(index, residues.length);
    residues.push({ ...residue, atoms: remaining });
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
    topology: {
      atoms,
      residues,
      chains,
      box: topology.box,
      conect: topology.conect
        .map(([a, b]) => [remap.get(a), remap.get(b)] as const)
        .filter((pair): pair is readonly [number, number] =>
          pair[0] !== undefined && pair[1] !== undefined,
        ),
      positions: Float64Array.from(positions),
    },
    keptIndices: kept,
  };
}

/** The two things OpenMM's preparation drops before rebuilding hydrogens. */
export function isWater(atom: PdbAtom): boolean {
  return atom.residueName === "HOH" || atom.residueName === "WAT";
}

export function isHydrogenAtom(atom: PdbAtom): boolean {
  return isHydrogen(atom);
}

/**
 * The variant for one residue, by OpenMM's rules.
 *
 * Returns `null` for "no specific variant", which is not the same as a missing
 * definition: a null variant still admits every hydrogen whose pH allows it,
 * which is how a neutral cysteine keeps its HG.
 */
export function chooseVariant(
  topology: PdbTopology,
  neighbours: readonly (readonly number[])[],
  residueIndex: number,
  ph: number,
): string | null {
  const residue = topology.residues[residueIndex] as PdbResidue;
  const atomsOf = (name: string): number[] =>
    residue.atoms.filter((a) => (topology.atoms[a] as PdbAtom).name === name);

  if (residue.name === "CYS") {
    // A sulfur bonded outside this residue is half a disulfide, so CYX --
    // regardless of pH.
    const sulfur = residue.atoms.filter(
      (a) => (topology.atoms[a] as PdbAtom).element.toUpperCase() === "S",
    );
    if (sulfur.length === 1) {
      const inside = new Set(residue.atoms);
      const bridged = (neighbours[sulfur[0] as number] as readonly number[]).some(
        (other) => !inside.has(other),
      );
      if (bridged) return "CYX";
    }
    return null;
  }

  if (residue.name === "HIS") {
    if (ph <= 6.5) return "HIP";
    const nd1 = atomsOf("ND1");
    const ne2 = atomsOf("NE2");
    if (nd1.length !== 1 || ne2.length !== 1) {
      throw new Error(
        `HIS residue ${residue.id} has ${nd1.length} ND1 and ${ne2.length} NE2 ` +
          "atoms; OpenMM refuses this too",
      );
    }
    const protonated = (atom: number): boolean =>
      (neighbours[atom] as readonly number[]).some((other) =>
        isHydrogen(topology.atoms[other] as PdbAtom),
      );
    const nd1Has = protonated(nd1[0] as number);
    const ne2Has = protonated(ne2[0] as number);
    if (nd1Has && ne2Has) return "HIP";
    if (nd1Has) return "HID";
    if (ne2Has) return "HIE";
    // 🔴 Always HID. See the module comment: OpenMM's hydrogen-bond comparison
    // cannot select HIE because `ne2IsBonded` is never assigned.
    return "HID";
  }

  return null;
}

/** Which hydrogens a residue should have, after all three filters. */
export function expectedHydrogens(
  spec: ResidueHydrogens,
  variant: string | null,
  ph: number,
  isNTerminal: boolean,
  isCTerminal: boolean,
  parentNames: ReadonlySet<string>,
): readonly HydrogenDef[] {
  return spec.hydrogens.filter((h) => {
    // Transcribed as three ors, which is not the same as "variant matches and
    // pH allows": with no specific variant chosen, a variant-tagged hydrogen is
    // admitted on pH alone, and that is what keeps HG on a neutral cysteine.
    const wanted =
      (variant === null && ph <= h.maxPh) ||
      (h.variants === null && ph <= h.maxPh) ||
      (h.variants !== null && variant !== null && h.variants.includes(variant));
    if (!wanted) return false;

    if (h.terminal !== null) {
      const allowed =
        (isNTerminal && h.terminal.includes("N")) ||
        (isCTerminal && h.terminal.includes("C")) ||
        (!isNTerminal && !isCTerminal && h.terminal.includes("-"));
      if (!allowed) return false;
    }
    return parentNames.has(h.parent);
  });
}

export interface AddHydrogensResult {
  readonly topology: PdbTopology;
  readonly bonds: readonly Pair[];
  /** Indices, in the new topology, of the atoms this added. */
  readonly added: readonly number[];
  /** The variant chosen per residue; `null` where none applies. */
  readonly variants: readonly (string | null)[];
}

/** 0.1 nm, the bond length OpenMM places a new hydrogen at. */
const HYDROGEN_OFFSET = 0.1;

type Point = readonly [number, number, number];

/**
 * Candidate directions on the unit sphere, by the Fibonacci spiral.
 *
 * A fixed, evenly spread set, computed once. Used to place hydrogens by search
 * rather than by construction -- see `placeHydrogen` for why that is the
 * approach -- and 512 points put neighbouring candidates about 9 degrees apart,
 * which is finer than the choice needs to be.
 */
const CANDIDATES: readonly Point[] = (() => {
  const count = 512;
  const golden = Math.PI * (3 - Math.sqrt(5));
  const points: Point[] = [];
  for (let i = 0; i < count; i++) {
    const z = 1 - (2 * i + 1) / count;
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    const theta = golden * i;
    points.push([radius * Math.cos(theta), radius * Math.sin(theta), z]);
  }
  return points;
})();

/**
 * Place one hydrogen 0.1 nm from its parent, as far as possible from every
 * bond the parent already has.
 *
 * 🔴 This replaces the obvious construction, which does not work. OpenMM sums
 * the vectors from each neighbour to the parent and offsets along that sum; for
 * a parent with one heavy neighbour the first hydrogen then lies exactly along
 * that bond's reverse, so adding it to the sum leaves the direction *unchanged*
 * and the second hydrogen lands on top of the first. Measured before the fix:
 * 92 coincident pairs across 46 parents on 1L2Y alone -- a methyl group as
 * three atoms at one point. OpenMM is saved from this twice over, by a random
 * jitter on each offset and by a `100/(r/0.1)^4` term in its placement
 * minimiser whose stated purpose is to spread hydrogens out. We do neither, and
 * coincident atoms are not a cosmetic problem: they are an infinite Coulomb and
 * Lennard-Jones energy that no minimiser recovers from.
 *
 * So the direction is chosen by maximising the smallest angle to the existing
 * bonds, over a fixed set of candidate directions. Greedy and sequential, so it
 * is not the globally optimal arrangement -- three hydrogens on a single heavy
 * neighbour come out near 90 degrees apart rather than the ideal 109.5 -- but it
 * is deterministic, needs no per-element case analysis, and cannot place two
 * atoms in the same spot. The user's minimiser does the rest, as it does for
 * OpenMM's random placement.
 */
function placeHydrogen(parent: Point, occupied: readonly Point[]): Point {
  if (occupied.length === 0) {
    return [parent[0] + HYDROGEN_OFFSET, parent[1], parent[2]];
  }

  // Unit vectors from the parent towards each thing already attached.
  const taken: Point[] = [];
  for (const other of occupied) {
    const dx = other[0] - parent[0];
    const dy = other[1] - parent[1];
    const dz = other[2] - parent[2];
    const length = Math.hypot(dx, dy, dz);
    if (length < 1e-12) continue;
    taken.push([dx / length, dy / length, dz / length]);
  }
  if (taken.length === 0) {
    return [parent[0] + HYDROGEN_OFFSET, parent[1], parent[2]];
  }

  let best = CANDIDATES[0] as Point;
  let bestScore = -Infinity;
  for (const candidate of CANDIDATES) {
    // The largest dot product is the smallest angle; minimise it.
    let worst = -Infinity;
    for (const direction of taken) {
      const dot =
        candidate[0] * direction[0] +
        candidate[1] * direction[1] +
        candidate[2] * direction[2];
      if (dot > worst) worst = dot;
    }
    const score = -worst;
    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }
  return [
    parent[0] + best[0] * HYDROGEN_OFFSET,
    parent[1] + best[1] * HYDROGEN_OFFSET,
    parent[2] + best[2] * HYDROGEN_OFFSET,
  ];
}

export function addHydrogens(
  topology: PdbTopology,
  bonds: readonly Pair[],
  definitions: HydrogenDefinitions,
  ph = 7,
): AddHydrogensResult {
  const neighbours: number[][] = topology.atoms.map(() => []);
  for (const [a, b] of bonds) {
    (neighbours[a] as number[]).push(b);
    (neighbours[b] as number[]).push(a);
  }

  const atoms: PdbAtom[] = [];
  const residues: { name: string; id: string; chainId: string; atoms: number[] }[] = [];
  const chains: { id: string; residues: number[] }[] = [];
  const positions: number[] = [];
  const added: number[] = [];
  const variants: (string | null)[] = topology.residues.map(() => null);
  /** Old index -> new index, for remapping the original bonds. */
  const remap = new Map<number, number>();
  const newBonds: Pair[] = [];

  for (const chain of topology.chains) {
    const newChain = { id: chain.id, residues: [] as number[] };
    chains.push(newChain);

    for (const [position, residueIndex] of chain.residues.entries()) {
      const residue = topology.residues[residueIndex] as PdbResidue;
      const newResidue = {
        name: residue.name,
        id: residue.id,
        chainId: residue.chainId,
        atoms: [] as number[],
      };
      const newResidueIndex = residues.length;
      residues.push(newResidue);
      newChain.residues.push(newResidueIndex);

      const spec = definitions.get(residue.name);
      const emit = (atom: PdbAtom, where: readonly [number, number, number]): number => {
        const index = atoms.length;
        atoms.push(atom);
        positions.push(where[0], where[1], where[2]);
        newResidue.atoms.push(index);
        return index;
      };
      const copy = (old: number): number => {
        const source = topology.atoms[old] as PdbAtom;
        const index = emit(source, [
          topology.positions[old * 3] as number,
          topology.positions[old * 3 + 1] as number,
          topology.positions[old * 3 + 2] as number,
        ]);
        remap.set(old, index);
        return index;
      };

      if (spec === undefined) {
        // Water, an ion, anything with no hydrogen definitions: copied through.
        for (const old of residue.atoms) copy(old);
        continue;
      }

      const isNTerminal = position === 0;
      const isCTerminal = position === chain.residues.length - 1;
      const variant = chooseVariant(topology, neighbours, residueIndex, ph);
      variants[residueIndex] = variant;

      const parentNames = new Set(
        residue.atoms
          .filter((a) => !isHydrogen(topology.atoms[a] as PdbAtom))
          .map((a) => (topology.atoms[a] as PdbAtom).name),
      );
      const wanted = expectedHydrogens(
        spec,
        variant,
        ph,
        isNTerminal,
        isCTerminal,
        parentNames,
      );

      for (const old of residue.atoms) {
        const source = topology.atoms[old] as PdbAtom;
        const newIndex = copy(old);
        if (isHydrogen(source)) continue;

        const existing = (neighbours[old] as readonly number[]).filter((other) =>
          isHydrogen(topology.atoms[other] as PdbAtom),
        );
        const expected = wanted.filter((h) => h.parent === source.name);
        // 🔴 Only when there are fewer than expected. A parent that already
        // carries as many hydrogens as the variant wants is left alone, even if
        // their names disagree -- OpenMM never renames or removes an existing
        // hydrogen unless the variant was given explicitly.
        if (existing.length >= expected.length) continue;

        // Match existing hydrogens to expected ones by name; the leftovers are
        // what gets added. Unmatched existing hydrogens consume an expected
        // slot from the end, which is OpenMM's arbitrary-but-defined choice.
        const unclaimed = [...expected];
        for (const hydrogen of existing) {
          const name = (topology.atoms[hydrogen] as PdbAtom).name;
          const match = unclaimed.findIndex((h) => h.name === name);
          if (match >= 0) unclaimed.splice(match, 1);
          else unclaimed.pop();
        }

        // The parent's existing bonds, as points, plus each hydrogen as it is
        // placed -- see the note in `placeHydrogen`.
        const parentPoint: Point = [
          topology.positions[old * 3] as number,
          topology.positions[old * 3 + 1] as number,
          topology.positions[old * 3 + 2] as number,
        ];
        const occupied: Point[] = (neighbours[old] as readonly number[]).map((other) => [
          topology.positions[other * 3] as number,
          topology.positions[other * 3 + 1] as number,
          topology.positions[other * 3 + 2] as number,
        ]);

        for (const definition of unclaimed) {
          const where = placeHydrogen(parentPoint, occupied);
          occupied.push(where);
          const index = emit(
            {
              serial: 0,
              name: definition.name,
              rawName: definition.name.padEnd(4).slice(0, 4),
              altLoc: "",
              residueName: residue.name,
              chainId: residue.chainId,
              residueId: residue.id,
              position: where,
              element: "H",
              hetatm: source.hetatm,
            },
            where,
          );
          added.push(index);
          newBonds.push([newIndex, index]);
        }
      }
    }
  }

  // The original bonds, remapped. Every atom is copied, so none is dropped --
  // but the lookup is still guarded, because `Modeller` does guard it and a
  // silent omission here would be a missing bond rather than an error.
  for (const [a, b] of bonds) {
    const first = remap.get(a);
    const second = remap.get(b);
    if (first !== undefined && second !== undefined) newBonds.push([first, second]);
  }

  return {
    topology: {
      atoms,
      residues,
      chains,
      box: topology.box,
      conect: [],
      positions: Float64Array.from(positions),
    },
    bonds: newBonds,
    added,
    variants,
  };
}
