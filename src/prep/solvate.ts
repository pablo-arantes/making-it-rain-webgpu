/**
 * Filling a periodic box with water and ions.
 *
 * Transcribed from `Modeller.addSolvent` and `Modeller._addIons`. The algorithm
 * is: pick a box, tile a pre-equilibrated water box through it, drop the waters
 * that clash with the solute, drop the ones that clash with each other across
 * the periodic boundary, then replace some waters with ions.
 *
 * Most of that is exactly reproducible and is verified against OpenMM position
 * by position. One step is not: `_addIons` chooses which waters to replace with
 * `random.choice`, so OpenMM does not reproduce its own ion placement between
 * runs. The *counts* follow from the solute charge and the water count and are
 * exact. See `docs/DEVIATIONS.md`.
 *
 * Three details that are easy to get wrong and are each worth a line:
 *
 * **The clash cutoff is per solute atom**, not one radius for everything:
 * `waterRadius + sigma * 0.5612310241546864907` for an atom with non-zero
 * epsilon, and `waterRadius` alone for one without. So solvation needs the
 * assigned Lennard-Jones parameters, which is why this takes an `Assignment`
 * rather than just a topology.
 *
 * **A water is kept or rejected on its oxygen alone.** The hydrogens follow it
 * wherever it goes, even outside the box. Testing all three atoms would reject
 * waters OpenMM keeps.
 *
 * **There are two separate clash tests.** The first is against the solute, by
 * summed radii. The second is between waters near opposite faces, which are far
 * apart in space but adjacent through the boundary — and it fires only when the
 * periodic distance is short *and* the direct distance is not, which is what
 * distinguishes "adjacent across the edge" from "genuinely neighbours".
 */

import type { PdbAtom, PdbResidue, PdbTopology } from "../analysis/pdb.js";
import { threefry2x32 } from "../reference/random.js";
import {
  type BoxVectors,
  boxVectorsForShape,
  isReducedForm,
  NotReducedForm,
  perpendicularWidths,
} from "../reference/triclinic.js";
import type { AssignedAtom } from "./assign.js";
import type { Pair } from "./bonded.js";

/** OpenMM's own constant: sigma to a van der Waals radius. */
export const VDW_RADIUS_PER_SIGMA = 0.5612310241546864907;

/** Water radii by model, as `addSolvent` hardcodes them (sigma, pre-scaling). */
const WATER_SIGMA: Readonly<Record<string, number>> = {
  tip3p: 0.31507524065751241,
  spce: 0.31657195050398818,
  tip4pew: 0.315365,
  tip5p: 0.312,
  swm4ndp: 0.318395,
};

/** Pure water is about 55.4 molar, which sets the ion count. */
export const WATER_MOLARITY = 55.4;

/** A counter-based domain, so ion choice cannot correlate with the thermostat. */
export const SOLVENT_DOMAIN = 0x3c6ef372;

/** The shapes `Modeller._computeBoxVectors` knows, and so the ones we know. */
export const KNOWN_BOX_SHAPES = ["cube", "dodecahedron", "octahedron"] as const;

export type BoxShape = (typeof KNOWN_BOX_SHAPES)[number];

export class UnsupportedBoxShape extends Error {
  constructor(shape: string) {
    super(
      `box shape "${shape}" is not one OpenMM knows. Known: ` +
        `${KNOWN_BOX_SHAPES.join(", ")}.`,
    );
    this.name = "UnsupportedBoxShape";
  }
}

/**
 * Cells an axis the engine's 27-cell shell needs, and so the box's real floor.
 *
 * 🔴 Mirrors `MINIMUM_CELLS_PER_AXIS` in `src/gpu/nonbonded.ts` **by value and
 * not by import**, so that the preparation page does not pull the kernels into
 * its bundle for one integer. `test/prep-solvate.test.ts` asserts the two are
 * equal, which is what keeps them from drifting.
 */
export const SHELL_CELLS_PER_AXIS = 3;

/**
 * Whether the engine will accept this cell, and what it would take.
 *
 * 🔴 Measured across the **perpendicular widths**, never the edges. A rhombic
 * dodecahedron's narrowest width is 0.707 of its edge, so it needs
 * `3*sqrt(2)*cutoff` -- 4.24 nm at a 1 nm cutoff against the cube's 3.0 -- and
 * 1L2Y at the notebook's 0.4 nm padding is refused where the cube is accepted.
 * Worth asking here rather than at run time: the alternative is a user
 * preparing a system, downloading it, and finding out on another page.
 *
 * The cutoff alone is the right radius to ask about even though the cell list
 * prefers cells of `cutoff + skin`: the engine falls back to cutoff-sized cells
 * when the wider ones would not give three an axis, and at exactly three the
 * shell spans the whole box, so nothing is out of reach.
 */
export function cellListFits(
  vectors: BoxVectors,
  cutoff: number,
): {
  readonly fits: boolean;
  /** nm, the narrowest perpendicular width. */
  readonly narrowest: number;
  /** nm, what that width has to reach. */
  readonly needed: number;
  /** nm, the box width that would satisfy it for this shape. */
  readonly width: number;
} {
  const narrowest = Math.min(...perpendicularWidths(vectors));
  const needed = SHELL_CELLS_PER_AXIS * cutoff;
  const edge = (vectors[0] as readonly number[])[0] as number;
  return {
    fits: narrowest >= needed,
    narrowest,
    needed,
    // The widths scale with the box, so the shape's own ratio gives the answer.
    width: (needed * edge) / narrowest,
  };
}

export interface SolvateOptions {
  /** nm. One of padding or boxSize. */
  readonly padding?: number;
  readonly boxSize?: readonly [number, number, number];
  readonly model?: string;
  /** molar, excluding the ions added to neutralise. */
  readonly ionicStrength?: number;
  readonly neutralize?: boolean;
  readonly positiveIon?: string;
  readonly negativeIon?: string;
  /** Seeds the ion placement, which OpenMM leaves to the interpreter's RNG. */
  readonly seed?: number;
  readonly boxShape?: string;
}

export interface SolvateResult {
  readonly topology: PdbTopology;
  readonly bonds: readonly Pair[];
  /**
   * 🔴 The full 3x3, because a diagonal cannot describe a dodecahedron and a
   * caller that built one from this field would get a rectangular cell of the
   * right volume and the wrong shape -- which no error would ever report.
   */
  readonly boxVectors: BoxVectors;
  readonly waters: number;
  readonly positiveIons: number;
  readonly negativeIons: number;
  /** The solute charge that was neutralised, rounded as OpenMM rounds it. */
  readonly soluteCharge: number;
}

const ION_ELEMENTS: Readonly<Record<string, string>> = {
  "Na+": "Na",
  "K+": "K",
  "Li+": "Li",
  "Rb+": "Rb",
  "Cs+": "Cs",
  "Cl-": "Cl",
  "Br-": "Br",
  "F-": "F",
  "I-": "I",
};

function elementOf(ion: string): string {
  const element = ION_ELEMENTS[ion];
  if (element === undefined) {
    throw new Error(`unknown ion "${ion}"; known: ${Object.keys(ION_ELEMENTS).join(", ")}`);
  }
  return element;
}

/**
 * A uniform grid over the periodic box, for "which of these points are near
 * that one".
 *
 * 🔴 Periodic, and that is not an optimisation. OpenMM builds its cell list
 * with `periodic=True`, so a water near one face is tested against solute atoms
 * near the opposite face — they are neighbours through the boundary. Searching
 * only non-periodic space under-rejects: with 1L2Y at the notebook's 4 Å
 * padding the box is 3.18 nm and the solute's bounding radius is 1.39 nm, so
 * the solute's own images sit right against every face. Measured: 855 waters
 * kept instead of 699, a 22% density error that nothing downstream would flag.
 *
 * The grid is aligned to the box rather than to the origin, with at least three
 * cells per axis so the 27-cell scan never wraps onto itself.
 */
/**
 * The neighbour grid `addSolvent` uses for its clash tests.
 *
 * 🔴 Bucketing is **rectangular, on positions folded into the cell**, and the
 * cells are sized from the box vectors' diagonal -- not from the perpendicular
 * widths the engine's cell list has to use. That is `_CellList` in OpenMM's
 * `modeller.py`, and copying it is the point rather than an economy: the water
 * this keeps has to be the water OpenMM keeps, so a better grid that survives
 * a different set of molecules is a worse grid here.
 *
 * The fold is the sequential triangular one -- subtract `c` by the floored
 * fractional z, then `b`, then `a` -- so a point lands in the rectangle the
 * diagonal spans. `near` then has to offset the *position* by a cell width and
 * fold again, rather than stepping the cell index as this did while the box was
 * always rectangular: stepping the index wraps z without the x and y shift that
 * crossing the `c` face carries, so a point by that face would be tested
 * against the wrong column and a clash there would go unseen. For a rectangular
 * cell the two are identical, which is why it only mattered once the box could
 * skew.
 *
 * Measured, by putting the index-stepping version back: a 1L2Y dodecahedron
 * keeps **488 waters where OpenMM keeps 483**, so five waters sit on top of the
 * solute's image across the `c` face. Nothing reports it -- the box is the
 * right shape and the right volume, the water is 1% too dense, and the run
 * starts normally. The cube is bit-identical under either version.
 */
class PeriodicGrid {
  private readonly cells = new Map<number, number[]>();
  private readonly counts: [number, number, number];
  private readonly sizes: [number, number, number];

  constructor(
    private readonly points: readonly (readonly [number, number, number])[],
    spacing: number,
    private readonly origin: readonly [number, number, number],
    private readonly vectors: BoxVectors,
  ) {
    const diagonal: [number, number, number] = [
      (vectors[0] as readonly number[])[0] as number,
      (vectors[1] as readonly number[])[1] as number,
      (vectors[2] as readonly number[])[2] as number,
    ];
    this.counts = [
      Math.max(3, Math.floor((diagonal[0] as number) / spacing)),
      Math.max(3, Math.floor((diagonal[1] as number) / spacing)),
      Math.max(3, Math.floor((diagonal[2] as number) / spacing)),
    ];
    this.sizes = [
      (diagonal[0] as number) / this.counts[0],
      (diagonal[1] as number) / this.counts[1],
      (diagonal[2] as number) / this.counts[2],
    ];
    for (const [index, point] of points.entries()) {
      const key = this.keyOf(this.cellOf(point));
      const existing = this.cells.get(key);
      if (existing === undefined) this.cells.set(key, [index]);
      else existing.push(index);
    }
  }

  /** Into the rectangle the diagonal spans, measured from the box's corner. */
  private fold(point: readonly [number, number, number]): [number, number, number] {
    const [a, b, c] = this.vectors;
    let x = (point[0] as number) - (this.origin[0] as number);
    let y = (point[1] as number) - (this.origin[1] as number);
    let z = (point[2] as number) - (this.origin[2] as number);
    const kc = Math.floor(z / (c[2] as number));
    x -= kc * (c[0] as number);
    y -= kc * (c[1] as number);
    z -= kc * (c[2] as number);
    const kb = Math.floor(y / (b[1] as number));
    x -= kb * (b[0] as number);
    y -= kb * (b[1] as number);
    x -= Math.floor(x / (a[0] as number)) * (a[0] as number);
    return [x, y, z];
  }

  private cellOf(point: readonly [number, number, number]): [number, number, number] {
    const folded = this.fold(point);
    const cell: [number, number, number] = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
      const count = this.counts[axis] as number;
      const raw = Math.floor((folded[axis] as number) / (this.sizes[axis] as number));
      cell[axis] = ((raw % count) + count) % count;
    }
    return cell;
  }

  private keyOf(cell: readonly [number, number, number]): number {
    return (
      (cell[0] as number) * (this.counts[1] as number) * (this.counts[2] as number) +
      (cell[1] as number) * (this.counts[2] as number) +
      (cell[2] as number)
    );
  }

  /** Indices in the 27 cells around a point, wrapping at the cell faces. */
  near(point: readonly [number, number, number]): number[] {
    const found: number[] = [];
    const seen = new Set<number>();
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const probe: readonly [number, number, number] = [
            (point[0] as number) + dx * (this.sizes[0] as number),
            (point[1] as number) + dy * (this.sizes[1] as number),
            (point[2] as number) + dz * (this.sizes[2] as number),
          ];
          const key = this.keyOf(this.cellOf(probe));
          if (seen.has(key)) continue;
          seen.add(key);
          const bucket = this.cells.get(key);
          if (bucket !== undefined) found.push(...bucket);
        }
      }
    }
    return found;
  }

  get size(): number {
    return this.points.length;
  }
}

/**
 * The squared distance between two points, under the periodic cell.
 *
 * 🔴 This is `minimumImage` in `src/reference/triclinic.ts` with the returned
 * array removed, because it runs once per candidate pair for every tiled water
 * -- millions of calls for a solvated box -- and `test/prep-solvate.test.ts`
 * holds the two to bit-for-bit agreement so the copy cannot drift.
 *
 * Measured against OpenMM's own `compiled.periodicDistance` on a 3 nm
 * dodecahedron: **0 of 20,000 separations differ**, worst 4.4e-16 nm. So this
 * is not merely *a* correct minimum image, it is the arithmetic `addSolvent`
 * tests clashes with, which is what makes the surviving water reproducible.
 *
 * The sequential reduction is not the true nearest image for every separation
 * -- over those same 20,000 pairs it reads up to 1.29 nm long -- and does not
 * need to be. It is exact below half the narrowest perpendicular width, and a
 * clash cutoff is about 0.33 nm against a bound of 1.06 nm for a 3 nm
 * dodecahedron, so every pair that could be a clash is reduced exactly.
 */
export function squaredPeriodicDistance(
  a: readonly [number, number, number],
  b: readonly [number, number, number],
  vectors: BoxVectors,
): number {
  const [va, vb, vc] = vectors;
  let x = (a[0] as number) - (b[0] as number);
  let y = (a[1] as number) - (b[1] as number);
  let z = (a[2] as number) - (b[2] as number);
  const kc = Math.round(z / (vc[2] as number));
  x -= kc * (vc[0] as number);
  y -= kc * (vc[1] as number);
  z -= kc * (vc[2] as number);
  const kb = Math.round(y / (vb[1] as number));
  x -= kb * (vb[0] as number);
  y -= kb * (vb[1] as number);
  x -= Math.round(x / (va[0] as number)) * (va[0] as number);
  return x * x + y * y + z * z;
}

/**
 * The box width for a padding, by OpenMM's rule.
 *
 * The bounding *sphere* about the centre of the bounding box, not the bounding
 * box itself — so the width is the sphere's diameter plus the padding, and at
 * least twice the padding so that no atom is within the padding of two images.
 */
export function boundingSphereRadius(positions: Float64Array, count: number): number {
  if (count === 0) return 0;
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let atom = 0; atom < count; atom++) {
    for (let axis = 0; axis < 3; axis++) {
      const value = positions[atom * 3 + axis] as number;
      if (value < (min[axis] as number)) min[axis] = value;
      if (value > (max[axis] as number)) max[axis] = value;
    }
  }
  const centre: [number, number, number] = [
    0.5 * (min[0] + max[0]),
    0.5 * (min[1] + max[1]),
    0.5 * (min[2] + max[2]),
  ];
  let radius = 0;
  for (let atom = 0; atom < count; atom++) {
    const dx = (positions[atom * 3] as number) - centre[0];
    const dy = (positions[atom * 3 + 1] as number) - centre[1];
    const dz = (positions[atom * 3 + 2] as number) - centre[2];
    radius = Math.max(radius, Math.hypot(dx, dy, dz));
  }
  return radius;
}

export function boxWidthForPadding(
  positions: Float64Array,
  count: number,
  padding: number,
): number {
  if (count === 0) return 2 * padding;
  const radius = boundingSphereRadius(positions, count);
  return Math.max(2 * radius + padding, 2 * padding);
}

interface WaterTemplate {
  /** Atom names in residue order, oxygen included. */
  readonly names: readonly string[];
  /** Offsets from the oxygen, nm. */
  readonly offsets: readonly (readonly [number, number, number])[];
  readonly oxygenIndex: number;
}

/** One water residue of the pre-equilibrated box, plus the box's own size. */
export function readWaterBox(topology: PdbTopology): {
  readonly box: readonly [number, number, number];
  readonly molecules: readonly { template: WaterTemplate; oxygen: readonly [number, number, number] }[];
} {
  if (topology.box === null) {
    throw new Error("the water box PDB has no CRYST1 record");
  }
  const molecules = topology.residues.map((residue) => {
    const names = residue.atoms.map((a) => (topology.atoms[a] as PdbAtom).name);
    const oxygenIndex = residue.atoms.findIndex(
      (a) => (topology.atoms[a] as PdbAtom).element.toUpperCase() === "O",
    );
    if (oxygenIndex < 0) throw new Error(`water residue ${residue.id} has no oxygen`);
    const oxygenAtom = residue.atoms[oxygenIndex] as number;
    const oxygen: readonly [number, number, number] = [
      topology.positions[oxygenAtom * 3] as number,
      topology.positions[oxygenAtom * 3 + 1] as number,
      topology.positions[oxygenAtom * 3 + 2] as number,
    ];
    const offsets = residue.atoms.map((a) => {
      return [
        (topology.positions[a * 3] as number) - oxygen[0],
        (topology.positions[a * 3 + 1] as number) - oxygen[1],
        (topology.positions[a * 3 + 2] as number) - oxygen[2],
      ] as const;
    });
    return { template: { names, offsets, oxygenIndex }, oxygen };
  });
  return { box: topology.box, molecules };
}

export function solvate(
  topology: PdbTopology,
  bonds: readonly Pair[],
  atoms: readonly AssignedAtom[],
  waterBox: PdbTopology,
  options: SolvateOptions = {},
): SolvateResult {
  const {
    padding,
    boxSize,
    model = "tip3p",
    ionicStrength = 0,
    neutralize = true,
    positiveIon = "Na+",
    negativeIon = "Cl-",
    seed = 0,
    boxShape = "cube",
  } = options;
  if (!(KNOWN_BOX_SHAPES as readonly string[]).includes(boxShape)) {
    throw new UnsupportedBoxShape(boxShape);
  }
  const shape = boxShape as BoxShape;

  const sigma = WATER_SIGMA[model];
  if (sigma === undefined) {
    throw new Error(
      `unknown water model "${model}"; known: ${Object.keys(WATER_SIGMA).join(", ")}`,
    );
  }
  const waterRadius = sigma * VDW_RADIUS_PER_SIGMA;

  // --- The box -----------------------------------------------------------
  // 🔴 `boxShape` applies to the padding branch and to nothing else, because
  // that is what `addSolvent` does: handed a box size, or a box on the
  // topology, OpenMM takes those vectors as they stand and never consults the
  // shape. Applying the shape to a size a caller gave explicitly would change
  // the volume they asked for by 30%.
  let vectors: BoxVectors;
  if (boxSize !== undefined) {
    vectors = [
      [boxSize[0] as number, 0, 0],
      [0, boxSize[1] as number, 0],
      [0, 0, boxSize[2] as number],
    ];
  } else if (padding !== undefined) {
    const width = boxWidthForPadding(topology.positions, topology.atoms.length, padding);
    vectors = boxVectorsForShape(shape, width);
  } else if (topology.box !== null) {
    vectors = [
      [topology.box[0] as number, 0, 0],
      [0, topology.box[1] as number, 0],
      [0, 0, topology.box[2] as number],
    ];
  } else {
    throw new Error(
      "neither a padding nor a box size was given, and the topology has no box",
    );
  }
  if (!isReducedForm(vectors)) throw new NotReducedForm(vectors);

  // 🔴 The water tiling, the inclusion test and the boundary skin all work in
  // the box vectors' **diagonal**, and that is not an approximation. For a
  // lower-triangular cell the rectangle `[0, ax) x [0, by) x [0, cz)` is a
  // fundamental domain of the lattice: its volume is the determinant, which is
  // the cell's own volume, and every point of the cell has exactly one image
  // in it. So filling that rectangle fills the cell, and `addSolvent` needs no
  // triclinic inclusion test anywhere -- only the clash distances and the
  // neighbour grid see the skew. A dodecahedron of width `w` is tiled over
  // `w x w x 0.707w`, which is why it holds 70.7% of the cube's water.
  const box: readonly [number, number, number] = [
    (vectors[0] as readonly number[])[0] as number,
    (vectors[1] as readonly number[])[1] as number,
    (vectors[2] as readonly number[])[2] as number,
  ];

  // --- Per-atom clash cutoffs -------------------------------------------
  const cutoff = atoms.map((atom) =>
    atom.epsilon !== 0 ? waterRadius + atom.sigma * VDW_RADIUS_PER_SIGMA : waterRadius,
  );
  const maxCutoff = Math.max(waterRadius, ...(cutoff.length > 0 ? cutoff : [waterRadius]));

  const solutePoints: (readonly [number, number, number])[] = [];
  for (let atom = 0; atom < topology.atoms.length; atom++) {
    solutePoints.push([
      topology.positions[atom * 3] as number,
      topology.positions[atom * 3 + 1] as number,
      topology.positions[atom * 3 + 2] as number,
    ]);
  }

  // The solute's bounding-box centre, which the water lattice is shifted onto.
  const centre: [number, number, number] = [0, 0, 0];
  if (solutePoints.length > 0) {
    for (let axis = 0; axis < 3; axis++) {
      let low = Infinity;
      let high = -Infinity;
      for (const point of solutePoints) {
        const value = point[axis] as number;
        if (value < low) low = value;
        if (value > high) high = value;
      }
      centre[axis] = (low + high) / 2;
    }
  }

  // The box's lower corner, which the grid is aligned to.
  const origin: readonly [number, number, number] = [
    centre[0] - (box[0] as number) / 2,
    centre[1] - (box[1] as number) / 2,
    centre[2] - (box[2] as number) / 2,
  ];
  const soluteGrid = new PeriodicGrid(solutePoints, maxCutoff, origin, vectors);
  const { box: tileBox, molecules } = readWaterBox(waterBox);

  // --- Tile, and keep what fits ------------------------------------------
  const tiles: [number, number, number] = [
    Math.ceil((box[0] as number) / (tileBox[0] as number)),
    Math.ceil((box[1] as number) / (tileBox[1] as number)),
    Math.ceil((box[2] as number) / (tileBox[2] as number)),
  ];
  const kept: { molecule: number; oxygen: [number, number, number] }[] = [];
  for (let tx = 0; tx < tiles[0]; tx++) {
    for (let ty = 0; ty < tiles[1]; ty++) {
      for (let tz = 0; tz < tiles[2]; tz++) {
        const offset: [number, number, number] = [
          tx * (tileBox[0] as number),
          ty * (tileBox[1] as number),
          tz * (tileBox[2] as number),
        ];
        for (const [index, molecule] of molecules.entries()) {
          const placed: [number, number, number] = [
            (molecule.oxygen[0] as number) + offset[0],
            (molecule.oxygen[1] as number) + offset[1],
            (molecule.oxygen[2] as number) + offset[2],
          ];
          // 🔴 Upper bound only, which is what `addSolvent` tests. The water
          // box's own coordinates are non-negative, so a lower test would be
          // redundant -- and adding one changes which waters survive.
          if (
            placed[0] > (box[0] as number) ||
            placed[1] > (box[1] as number) ||
            placed[2] > (box[2] as number)
          ) {
            continue;
          }
          const shifted: [number, number, number] = [
            placed[0] + centre[0] - (box[0] as number) / 2,
            placed[1] + centre[1] - (box[1] as number) / 2,
            placed[2] + centre[2] - (box[2] as number) / 2,
          ];
          let clashes = false;
          for (const other of soluteGrid.near(shifted)) {
            const limit = cutoff[other] as number;
            if (
              squaredPeriodicDistance(shifted, solutePoints[other] as readonly [number, number, number], vectors) <
              limit * limit
            ) {
              clashes = true;
              break;
            }
          }
          if (!clashes) kept.push({ molecule: index, oxygen: shifted });
        }
      }
    }
  }

  // --- Drop waters that only clash through the boundary -------------------
  const upper: [number, number, number] = [
    centre[0] + (box[0] as number) / 2 - waterRadius,
    centre[1] + (box[1] as number) / 2 - waterRadius,
    centre[2] + (box[2] as number) / 2 - waterRadius,
  ];
  const lower: [number, number, number] = [
    centre[0] - (box[0] as number) / 2 + waterRadius,
    centre[1] - (box[1] as number) / 2 + waterRadius,
    centre[2] - (box[2] as number) / 2 + waterRadius,
  ];
  const skin = kept
    .filter(
      ({ oxygen }) =>
        oxygen[0] < lower[0] || oxygen[1] < lower[1] || oxygen[2] < lower[2],
    )
    .map(({ oxygen }) => oxygen as readonly [number, number, number]);
  const skinGrid = new PeriodicGrid(skin, maxCutoff, origin, vectors);

  const survivors = kept.filter(({ oxygen }) => {
    if (oxygen[0] < upper[0] && oxygen[1] < upper[1] && oxygen[2] < upper[2]) {
      return true;
    }
    for (const index of skinGrid.near(oxygen)) {
      const other = skin[index] as readonly [number, number, number];
      const periodic = squaredPeriodicDistance(other, oxygen, vectors);
      const direct =
        ((other[0] as number) - oxygen[0]) ** 2 +
        ((other[1] as number) - oxygen[1]) ** 2 +
        ((other[2] as number) - oxygen[2]) ** 2;
      // Close through the boundary but not directly: the pair only exists
      // because of the wrap, so one of them has to go.
      if (periodic < waterRadius * waterRadius && direct > waterRadius * waterRadius) {
        return false;
      }
    }
    return true;
  });

  // --- Ion counts --------------------------------------------------------
  let charge = 0;
  for (const atom of atoms) charge += atom.charge;
  const soluteCharge = Math.floor(0.5 + charge);
  let positives = 0;
  let negatives = 0;
  if (neutralize) {
    if (Math.abs(soluteCharge) > survivors.length) {
      throw new Error(
        `cannot neutralise a charge of ${soluteCharge} with only ` +
          `${survivors.length} waters to replace`,
      );
    }
    if (soluteCharge > 0) negatives += soluteCharge;
    else positives -= soluteCharge;
  }
  if (ionicStrength > 0) {
    const count =
      ((survivors.length - positives - negatives) * ionicStrength) / WATER_MOLARITY;
    const pairs = Math.floor(count + 0.5);
    positives += pairs;
    negatives += pairs;
  }

  // 🔴 Which waters become ions is a deviation. OpenMM uses `random.choice`,
  // so it does not reproduce its own placement; we draw from Threefry on
  // (index, seed) so that ours is reproducible from `run.json`. The counts
  // above are exact either way. The 0.05 nm separation test is OpenMM's.
  const replaced = new Set<number>();
  const chosen: number[] = [];
  const ionCutoff = 0.05;
  let draw = 0;
  while (chosen.length < positives + negatives) {
    if (draw > 1000 * (positives + negatives) + 1000) {
      throw new Error(
        `could not place ${positives + negatives} ions with a ${ionCutoff} nm ` +
          "separation; the box is too crowded",
      );
    }
    const [word] = threefry2x32([draw, SOLVENT_DOMAIN], [seed, 0]);
    draw++;
    const candidate = (word >>> 0) % survivors.length;
    if (replaced.has(candidate)) continue;
    const here = (survivors[candidate] as { oxygen: [number, number, number] }).oxygen;
    let tooClose = false;
    for (const other of chosen) {
      const there = (survivors[other] as { oxygen: [number, number, number] }).oxygen;
      if (
        Math.hypot(here[0] - there[0], here[1] - there[1], here[2] - there[2]) <= ionCutoff
      ) {
        tooClose = true;
        break;
      }
    }
    if (tooClose) continue;
    replaced.add(candidate);
    chosen.push(candidate);
  }

  // --- Assemble ----------------------------------------------------------
  const newAtoms: PdbAtom[] = [...topology.atoms];
  const residues: PdbResidue[] = topology.residues.map((r) => ({ ...r }));
  const chains = topology.chains.map((c) => ({ id: c.id, residues: [...c.residues] }));
  const positions: number[] = [...topology.positions];
  const newBonds: Pair[] = [...bonds];

  const waterChain = { id: chains.length === 0 ? "A" : "", residues: [] as number[] };
  chains.push(waterChain);
  for (const [index, survivor] of survivors.entries()) {
    if (replaced.has(index)) continue;
    const template = (molecules[survivor.molecule] as { template: WaterTemplate }).template;
    const residueIndex = residues.length;
    const residueAtoms: number[] = [];
    let oxygenAtom = -1;
    for (const [slot, name] of template.names.entries()) {
      const offset = template.offsets[slot] as readonly [number, number, number];
      const atomIndex = newAtoms.length;
      const where: readonly [number, number, number] = [
        survivor.oxygen[0] + (offset[0] as number),
        survivor.oxygen[1] + (offset[1] as number),
        survivor.oxygen[2] + (offset[2] as number),
      ];
      newAtoms.push({
        serial: 0,
        name,
        rawName: name.padEnd(4).slice(0, 4),
        altLoc: "",
        residueName: "HOH",
        chainId: waterChain.id,
        residueId: String(residueIndex + 1),
        position: where,
        element: slot === template.oxygenIndex ? "O" : "H",
        hetatm: false,
      });
      positions.push(where[0], where[1], where[2]);
      residueAtoms.push(atomIndex);
      if (slot === template.oxygenIndex) oxygenAtom = atomIndex;
    }
    for (const atom of residueAtoms) {
      if (atom !== oxygenAtom) newBonds.push([oxygenAtom, atom]);
    }
    residues.push({ name: "HOH", id: String(residueIndex + 1), chainId: waterChain.id, atoms: residueAtoms });
    waterChain.residues.push(residueIndex);
  }

  const ionChain = { id: "", residues: [] as number[] };
  chains.push(ionChain);
  for (const [slot, index] of chosen.entries()) {
    const ion = slot < positives ? positiveIon : negativeIon;
    const element = elementOf(ion);
    const where = (survivors[index] as { oxygen: [number, number, number] }).oxygen;
    const residueIndex = residues.length;
    const atomIndex = newAtoms.length;
    newAtoms.push({
      serial: 0,
      name: element,
      rawName: element.padEnd(4).slice(0, 4),
      altLoc: "",
      residueName: element.toUpperCase(),
      chainId: ionChain.id,
      residueId: String(residueIndex + 1),
      position: where,
      element,
      hetatm: false,
    });
    positions.push(where[0], where[1], where[2]);
    residues.push({
      name: element.toUpperCase(),
      id: String(residueIndex + 1),
      chainId: ionChain.id,
      atoms: [atomIndex],
    });
    ionChain.residues.push(residueIndex);
  }

  return {
    topology: {
      atoms: newAtoms,
      residues,
      chains,
      box,
      conect: topology.conect,
      positions: Float64Array.from(positions),
    },
    bonds: newBonds,
    boxVectors: vectors,
    waters: survivors.length - chosen.length,
    positiveIons: positives,
    negativeIons: negatives,
    soluteCharge,
  };
}
