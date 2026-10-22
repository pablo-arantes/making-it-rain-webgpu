/**
 * Host side of the real-space nonbonded kernels.
 *
 * Builds the cell list and the per-atom exclusion lists, runs the two
 * dispatches into one shared force and energy accumulator, and adds the
 * analytic dispersion correction, which depends on the volume but on no
 * coordinate and so is a host-side constant.
 *
 * The cell list is built on the device (src/gpu/cell-list.ts) and written into
 * sub-ranges of the same `lists` buffer the exclusion lists live in. They share
 * a buffer because the pair kernel reads all four arrays through one storage
 * binding, and splitting them would put it over the eight storage buffers
 * WebGPU guarantees per stage. `listSections` is the single description of that
 * layout; both the cell-list passes and the kernel's offsets come from it.
 *
 * Like the bonded path, this is split into `prepareDirect` (everything that
 * depends on the `System` and not the coordinates), `planDirect` (the passes)
 * and `evaluateDirectSpace` (one configuration, for the differential tests), so
 * a step loop and the tests cannot drift apart.
 */

import { pipelineCacheFor, readBuffer } from "./compute.js";
import { submitPasses, type Pass, type PipelineCache } from "./pipeline.js";
import { BufferPool } from "./allocator.js";
import { readableStorageUsage, storageUsage, uniformUsage } from "./usage.js";
import {
  alignedBytes,
  gridExtents,
  planCellList,
  planWrapPositions,
  prepareCellList,
  type CellListBuffers,
  type ListTarget,
} from "./cell-list.js";
import { NONBONDED_EXCEPTIONS, NONBONDED_PAIRS, VERLET } from "./kernels.js";
import { unpackFixed } from "./forces.js";
import { fixedWords, fromFixed, fromWords } from "../reference/fixed-point.js";
import {
  dispersionCorrection,
  rectangularBox,
} from "../reference/forces/nonbonded.js";
import {
  fractionalOf,
  inverseBox,
  isRectangular,
  perpendicularWidths,
  type BoxVectors,
} from "../reference/triclinic.js";
import type { NonbondedForce, Vec3 } from "../engine/system.js";

/** Three or more cells an axis, or the 27-cell scan would visit one twice. */
export const MINIMUM_CELLS_PER_AXIS = 3;

export class CellListTooCoarse extends Error {
  constructor(readonly cells: Vec3, readonly box: Vec3, readonly cutoff: number) {
    super(
      `a ${box.map((v) => v.toFixed(3)).join(" x ")} nm box with a ${cutoff} nm ` +
        `cutoff gives only ${cells.join(" x ")} cells, and the 27-cell scan ` +
        `needs at least ${MINIMUM_CELLS_PER_AXIS} an axis. The box is less than ` +
        `three cutoffs wide, which also means the minimum image convention is ` +
        `close to breaking: see docs/NOTEBOOK_PARITY.md finding 4.`,
    );
    this.name = "CellListTooCoarse";
  }
}

export interface CellList {
  readonly cells: Vec3;
  readonly cellCount: number;
  /** cellCount + 1 prefix sums, then the sorted atom order. */
  readonly starts: Uint32Array;
  readonly order: Uint32Array;
}

export function buildCellList(
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  cutoff: number,
): CellList {
  const particleCount = positions.length / 3;
  const vectors = boxVectors as unknown as BoxVectors;
  const skewed = !isRectangular(vectors);
  // 🔴 Cells are counted against the *perpendicular* widths, not the edge
  // lengths. A 27-cell scan reaches one cell width, so sizing a skewed cell's
  // slabs by an edge would make them thinner than the cutoff across the faces
  // and lose pairs. For a rectangular cell the two are the same number.
  const widths = skewed
    ? (perpendicularWidths(vectors) as unknown as Vec3)
    : ([
        (boxVectors[0] as Vec3)[0] as number,
        (boxVectors[1] as Vec3)[1] as number,
        (boxVectors[2] as Vec3)[2] as number,
      ] as Vec3);
  const cells: Vec3 = [
    Math.max(1, Math.floor((widths[0] as number) / cutoff)),
    Math.max(1, Math.floor((widths[1] as number) / cutoff)),
    Math.max(1, Math.floor((widths[2] as number) / cutoff)),
  ];
  if (cells.some((n) => n < MINIMUM_CELLS_PER_AXIS)) {
    throw new CellListTooCoarse(cells, widths, cutoff);
  }
  const cellCount = (cells[0] as number) * (cells[1] as number) * (cells[2] as number);
  const cellOf = new Int32Array(particleCount);
  const starts = new Uint32Array(cellCount + 1);
  const inverse = skewed ? inverseBox(vectors) : undefined;
  const bucket = (fraction: number, divisions: number): number => {
    const wrapped = fraction - Math.floor(fraction);
    const i = Math.floor(wrapped * divisions);
    return i === divisions ? divisions - 1 : i;
  };
  for (let i = 0; i < particleCount; i++) {
    const x = positions[i * 3] as number;
    const y = positions[i * 3 + 1] as number;
    const z = positions[i * 3 + 2] as number;
    let cx: number;
    let cy: number;
    let cz: number;
    if (inverse === undefined) {
      cx = bucket(x / (widths[0] as number), cells[0] as number);
      cy = bucket(y / (widths[1] as number), cells[1] as number);
      cz = bucket(z / (widths[2] as number), cells[2] as number);
    } else {
      // Fractional coordinates: a cell is a parallelepiped of the box, and
      // `f - floor(f)` wraps it however skewed it is.
      const fractional = fractionalOf(x, y, z, inverse);
      cx = bucket(fractional[0] as number, cells[0] as number);
      cy = bucket(fractional[1] as number, cells[1] as number);
      cz = bucket(fractional[2] as number, cells[2] as number);
    }
    const cell = (cx * (cells[1] as number) + cy) * (cells[2] as number) + cz;
    cellOf[i] = cell;
    starts[cell + 1] = (starts[cell + 1] as number) + 1;
  }
  for (let c = 0; c < cellCount; c++) {
    starts[c + 1] = (starts[c + 1] as number) + (starts[c] as number);
  }
  const order = new Uint32Array(particleCount);
  const cursor = Uint32Array.from(starts.subarray(0, cellCount));
  for (let i = 0; i < particleCount; i++) {
    const cell = cellOf[i] as number;
    order[cursor[cell] as number] = i;
    cursor[cell] = (cursor[cell] as number) + 1;
  }
  return { cells, cellCount, starts, order };
}

export interface ExclusionLists {
  /** particleCount + 1 prefix sums into `list`. */
  readonly starts: Uint32Array;
  readonly list: Uint32Array;
}

/**
 * Each atom's exclusions, both ways round.
 *
 * The gather kernel asks "is j excluded from i?" from i's point of view only,
 * so every exception has to appear in both atoms' lists.
 */
export function buildExclusionLists(force: NonbondedForce): ExclusionLists {
  const n = force.particleCount;
  const starts = new Uint32Array(n + 1);
  for (let e = 0; e < force.exceptionCount; e++) {
    starts[(force.exceptionP1[e] as number) + 1] =
      (starts[(force.exceptionP1[e] as number) + 1] as number) + 1;
    starts[(force.exceptionP2[e] as number) + 1] =
      (starts[(force.exceptionP2[e] as number) + 1] as number) + 1;
  }
  for (let i = 0; i < n; i++) {
    starts[i + 1] = (starts[i + 1] as number) + (starts[i] as number);
  }
  const list = new Uint32Array(force.exceptionCount * 2);
  const cursor = Uint32Array.from(starts.subarray(0, n));
  for (let e = 0; e < force.exceptionCount; e++) {
    const a = force.exceptionP1[e] as number;
    const b = force.exceptionP2[e] as number;
    list[cursor[a] as number] = b;
    cursor[a] = (cursor[a] as number) + 1;
    list[cursor[b] as number] = a;
    cursor[b] = (cursor[b] as number) + 1;
  }
  return { starts, list };
}

export interface DirectSpaceResult {
  /** The pair kernel's contribution, kJ/mol. */
  readonly pairEnergy: number;
  /** The exclusion corrections and exceptions, kJ/mol. */
  readonly exceptionEnergy: number;
  /** The analytic long-range correction, computed on the host. */
  readonly dispersion: number;
  readonly energy: number;
  /** kJ/mol/nm, xyz interleaved. */
  readonly forces: Float64Array;
}

/**
 * Where each array sits inside the shared `lists` buffer.
 *
 * Byte offsets, each a multiple of 256 so that the cell-list kernel can bind
 * its three arrays as sub-ranges -- 256 is WebGPU's default
 * `minStorageBufferOffsetAlignment`. The pair kernel instead binds the whole
 * buffer and indexes it with the word offsets below, which is why the padding
 * costs nothing: it is a handful of KB and it means neither side has to know
 * the other's packing.
 */
/**
 * The default Verlet skin, nm.
 *
 * The list is built to `cutoff + skin` and stays valid while no atom has moved
 * `skin / 2`, so the skin trades memory and per-step candidates against how
 * often the shell scan is paid. Measured on 3HTB at 26,749 atoms, against the
 * 82 steps/s the cell-shell path manages on the same run:
 *
 *   skin   steps/s   steps/rebuild   list bytes
 *   0.05     128.0        1.4           81.5 MB
 *   0.10     135.2        2.8           93.2 MB
 *   0.15     151.3        4.3          105.9 MB
 *   0.20     156.6        5.6          119.8 MB
 *
 * 0.15 is where the curve has mostly flattened and the list still fits inside
 * the 128 MB storage binding WebGPU guarantees, which 0.20 is within 8 MB of.
 * The rebuild interval is short because the criterion is exact -- the list is
 * rebuilt when *any* atom has moved half the skin, and the fastest hydrogen in
 * 26,749 atoms moves fast. GROMACS gets longer intervals by sizing the skin
 * from a tolerated energy drift instead, which is a different promise.
 */
export const DEFAULT_VERLET_SKIN = 0.15;

/**
 * How many neighbours to make room for per atom.
 *
 * The mean is the number density times the volume of the list's sphere. The
 * margin is for the fluctuation around it and for the solute, which is denser
 * than the water it sits in: the count over a sphere holding `m` atoms varies
 * by about `sqrt(m)`, which is 4% at 520, and a buried protein atom sees
 * roughly 1.3x the mean. 1.6 covers both with room over.
 *
 * Overflow past this is not an error and not silent: `build_neighbours` marks
 * the atom and `direct_pairs` evaluates it from the cell shell instead, so the
 * margin being wrong costs speed rather than physics.
 */
export const VERLET_CAPACITY_MARGIN = 1.6;

export function verletCapacity(
  particleCount: number,
  box: Vec3,
  cutoff: number,
  skin: number,
): number {
  const volume = (box[0] as number) * (box[1] as number) * (box[2] as number);
  const radius = cutoff + skin;
  const mean = ((4 / 3) * Math.PI * radius ** 3 * particleCount) / volume;
  // Never smaller than the system, which is the most an atom can ever see, and
  // never zero for a system small enough to round down to it.
  return Math.max(1, Math.min(particleCount - 1, Math.ceil(mean * VERLET_CAPACITY_MARGIN)));
}

/** How many invocations share one atom's list; see `direct_pairs_list`. */
export const VERLET_SLICES = 8;

export interface ListSections {
  readonly cellStarts: number;
  readonly order: number;
  readonly exclusionStarts: number;
  readonly exclusionList: number;
  readonly cellOf: number;
  readonly totalBytes: number;
}

export function listSections(
  cellCount: number,
  particleCount: number,
  exclusionListLength: number,
): ListSections {
  const cellStarts = 0;
  const order = cellStarts + alignedBytes((cellCount + 1) * 4);
  const exclusionStarts = order + alignedBytes(particleCount * 4);
  const exclusionList = exclusionStarts + alignedBytes((particleCount + 1) * 4);
  const cellOf = exclusionList + alignedBytes(Math.max(1, exclusionListLength) * 4);
  return {
    cellStarts,
    order,
    exclusionStarts,
    exclusionList,
    cellOf,
    totalBytes: cellOf + alignedBytes(particleCount * 4),
  };
}

/**
 * Write the box and half the box, per axis, in fixed point.
 *
 * Six `vec4<u32>` at `words[offset]`, of which only the first two words of each
 * are used: the box at `offset`, half the box 12 words later. WGSL gives a
 * uniform array element a 16-byte stride, so the padding is part of the layout
 * rather than slack.
 *
 * Half is passed to the kernel rather than derived there because a 64-bit shift
 * is more WGSL than a second uniform field.
 */
export function writeFixedBox(words: Uint32Array, offset: number, box: Vec3): void {
  for (let axis = 0; axis < 3; axis++) {
    const extent = box[axis] as number;
    const whole = fixedWords(extent);
    words[offset + axis * 4] = whole.lo;
    words[offset + axis * 4 + 1] = whole.hi;
    const half = fixedWords(extent / 2);
    words[offset + 12 + axis * 4] = half.lo;
    words[offset + 12 + axis * 4 + 1] = half.hi;
  }
}

/**
 * The three off-diagonal lattice components, in fixed point: `b.x`, `c.x`,
 * `c.y`.
 *
 * 🔴 All a general cell adds. The diagonal is already in `writeFixedBox`, and a
 * rectangular cell has nothing else — so these are zero for every fixture this
 * project was built on, which is what makes the conversion verifiable by
 * "rectangular unchanged".
 */
export function writeOffDiagonalBox(
  words: Uint32Array,
  offset: number,
  boxVectors: readonly [Vec3, Vec3, Vec3],
): void {
  const components = [
    (boxVectors[1] as Vec3)[0] as number,
    (boxVectors[2] as Vec3)[0] as number,
    (boxVectors[2] as Vec3)[1] as number,
  ];
  for (const [slot, value] of components.entries()) {
    const packed = fixedWords(value);
    words[offset + slot * 4] = packed.lo;
    words[offset + slot * 4 + 1] = packed.hi;
  }
}

/**
 * The inverse box, as three `vec3<f32>` rows at a 16-byte stride.
 *
 * f32 is deliberate: this only ever *decides* which image to take, never
 * carries a result. See `nearest_image_triclinic`.
 */
export function writeInverseBox(
  floats: Float32Array,
  offset: number,
  boxVectors: readonly [Vec3, Vec3, Vec3],
): void {
  const inverse = inverseBox(boxVectors as unknown as BoxVectors);
  for (let row = 0; row < 3; row++) {
    for (let column = 0; column < 3; column++) {
      floats[offset + row * 4 + column] = (inverse[row] as Vec3)[column] as number;
    }
  }
}

/**
 * 256 bytes. Five u32 word offsets fill 0..19; `cells` takes its 16-byte
 * boundary at 32 and the cutoff follows at 44; the f32 box takes the next one
 * at 48 with alpha at 60; the fixed-point diagonal at 64 and half of it at 112;
 * then the three off-diagonals at 160 and the inverse box at 208, three
 * `vec4`-strided rows each.
 */
export const PAIR_PARAMS_BYTES = 256;

/** 32 bytes of struct, then the fixed-point box and half box to 128. */
export const EXCEPTION_PARAMS_BYTES = 128;

/**
 * Pack the pair kernel's uniform.
 *
 * A function rather than inline code because the barostat rewrites it: the box
 * is no longer a constant of the run. The cell *divisions* stay fixed -- see
 * `StepLoop.setBox` for why -- so only the extents implied by them change.
 */
export function packPairParams(
  force: NonbondedForce,
  sections: ListSections,
  divisions: Vec3,
  box: Vec3,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  verlet: { capacity: number; skin: number } = { capacity: 0, skin: 0 },
): Uint8Array {
  const bytes = new ArrayBuffer(PAIR_PARAMS_BYTES);
  const u32 = new Uint32Array(bytes);
  const f32 = new Float32Array(bytes);
  u32[0] = force.particleCount;
  u32[1] = sections.order / 4;
  u32[2] = sections.exclusionStarts / 4;
  u32[3] = sections.exclusionList / 4;
  u32[4] = sections.cellOf / 4;
  // Words 5 to 7 were padding before `cells`, which is 16-byte aligned.
  u32[5] = verlet.capacity;
  u32[6] = VERLET_SLICES;
  f32[7] = (force.cutoff + verlet.skin) ** 2;
  u32[8] = divisions[0] as number;
  u32[9] = divisions[1] as number;
  u32[10] = divisions[2] as number;
  f32[11] = force.cutoff;
  f32[12] = box[0] as number;
  f32[13] = box[1] as number;
  f32[14] = box[2] as number;
  f32[15] = alpha;
  writeFixedBox(u32, 16, box);
  writeOffDiagonalBox(u32, 40, boxVectors);
  writeInverseBox(f32, 52, boxVectors);
  return new Uint8Array(bytes);
}

/** Pack the exception kernel's uniform. */
export function packExceptionParams(
  force: NonbondedForce,
  box: Vec3,
  alpha: number,
): Uint8Array {
  const bytes = new ArrayBuffer(EXCEPTION_PARAMS_BYTES);
  const u32 = new Uint32Array(bytes);
  const f32 = new Float32Array(bytes);
  u32[0] = force.exceptionCount;
  f32[1] = alpha;
  f32[4] = box[0] as number;
  f32[5] = box[1] as number;
  f32[6] = box[2] as number;
  u32[7] = force.exceptionCount;
  writeFixedBox(u32, 8, box);
  return new Uint8Array(bytes);
}

/** Everything the direct-space kernels need that the coordinates do not change. */
export interface ResidentDirect {
  readonly force: NonbondedForce;
  readonly particleCount: number;
  readonly box: Vec3;
  readonly alpha: number;
  readonly sections: ListSections;
  readonly lists: GPUBuffer;
  readonly particle: GPUBuffer;
  readonly pairParams: GPUBuffer;
  readonly exceptions: GPUBuffer;
  readonly exceptionParams: GPUBuffer;
  readonly cellList: CellListBuffers;
  /** kJ/mol, analytic, a function of the volume and no coordinate. */
  readonly dispersion: number;
  readonly verlet: ResidentVerlet;
}

/**
 * The Verlet neighbour list and what decides when to rebuild it.
 *
 * `capacity` of 0 means no list: `planDirect` then plans the cell-shell path
 * for every atom and none of these buffers are read. The differential tests use
 * that, so the two candidate paths can be compared on the same configuration.
 */
export interface ResidentVerlet {
  readonly capacity: number;
  readonly skin: number;
  /**
   * The radius the cell grid was sized by, which is not always `cutoff + skin`.
   *
   * 🔴 Recorded rather than recomputed, because it is clamped: a box too small
   * to divide by `cutoff + skin` and still give three cells an axis uses the
   * cutoff instead, and relies on three cells meaning the 27-cell shell is the
   * whole box. `StepLoop.setBox` re-derived it as `cutoff + skin` and threw on
   * 1L2Y the moment the barostat moved the box -- 1.027 nm cells against a
   * 1.15 nm radius -- which the bonded browser test caught.
   */
  readonly searchRadius: number;
  /** `[k * particleCount + atom]`; see the binding comment in the kernel. */
  readonly neighbours: GPUBuffer;
  readonly counts: GPUBuffer;
  /** `[shell * particleCount + atom]`, counts then offsets. */
  readonly shellCounts: GPUBuffer;
  /** Positions when the list was built, for the displacement test. */
  readonly reference: GPUBuffer;
  /** rebuild, built, overflow. */
  readonly flags: GPUBuffer;
  readonly params: GPUBuffer;
  readonly byteLength: number;
}

export function prepareDirect(
  pool: BufferPool,
  force: NonbondedForce,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  /** 0 builds no list, so every atom takes the cell-shell path. */
  verletSkin?: number,
): ResidentDirect {
  const device = pool.device;
  const box = rectangularBox(boxVectors);
  const n = force.particleCount;
  const exclusions = buildExclusionLists(force);

  // 🔴 The list has to fit one storage binding, and at a water density it is
  // about 4 kB an atom -- which is 33,000 atoms inside the 128 MB WebGPU
  // guarantees. So it is chosen to fit, not assumed to: the skin comes down
  // first, and if even a thin one will not fit, the list is dropped and every
  // atom takes the cell-shell path. That path is not a degraded mode kept for
  // this -- it is the same kernel the differential tests run, and
  // `validateNeighbours` holds the two to bit-for-bit agreement -- so giving
  // way costs speed and nothing else.
  const bindingBudget = Math.min(
    pool.device.limits.maxStorageBufferBindingSize,
    pool.device.limits.maxBufferSize,
  );
  const fitted = ((): { skin: number; capacity: number } => {
    if (verletSkin === 0) return { skin: 0, capacity: 0 };
    const wanted = verletSkin ?? DEFAULT_VERLET_SKIN;
    for (const candidate of [wanted, 0.1, 0.05, 0.02]) {
      if (candidate > wanted) continue;
      const room = verletCapacity(n, box, force.cutoff, candidate);
      if (room * n * 4 <= bindingBudget) return { skin: candidate, capacity: room };
    }
    return { skin: 0, capacity: 0 };
  })();
  const skin = fitted.skin;
  const capacity = fitted.capacity;
  /**
   * The radius the cells are sized by, so the 27-cell shell reaches it.
   *
   * 🔴 Sized by `cutoff + skin`, not by the cutoff, and the difference is a
   * lost-pair bug rather than a tuning choice. A shell of one cell either way
   * covers everything within *one cell width*, so a search radius larger than
   * the cell leaves candidates two cells out that the scan never visits. With
   * cells sized by the cutoff, 3HTB's are 1.096 nm and a list built to
   * `1.0 + 0.1` reaches 1.1 -- so every pair between 1.096 and 1.1 nm was
   * missing, and the trajectory came apart a few hundred steps later.
   *
   * The exception is a box too small to divide by the larger radius and still
   * get three cells an axis, which is 1L2Y: 3.081 nm over 1.1 is 2.8. Three
   * cells an axis is the floor, and at exactly three the shell visits every
   * cell in the box -- so nothing can be out of its reach whatever the radius,
   * and the cutoff-sized grid is both correct and finer. `cellGeometry` throws
   * if even the cutoff cannot give three, which is the pre-existing minimum
   * image requirement and not this one.
   *
   * 1L2Y being that case is also why it hid the bug: its differential checks
   * pass with any search radius, so the small fixture cannot test this at all.
   * `validateNeighbours` is the check that can, on the big one.
   */
  // 🔴 The narrowest *perpendicular* width, not the shortest edge: it is what
  // the 27-cell shell has to reach across, and for a skewed cell it is smaller.
  const shortestBox = Math.min(
    ...(gridExtents(boxVectors) as unknown as number[]),
  );
  const searchRadius =
    capacity === 0 ||
    Math.floor(shortestBox / (force.cutoff + skin)) < MINIMUM_CELLS_PER_AXIS
      ? force.cutoff
      : force.cutoff + skin;

  // The grid has to be known before the sections can be laid out.
  const probe = prepareCellList(pool, n, boxVectors, searchRadius, {
    buffer: pool.buffer({ label: "nonbonded/listsProbe", byteLength: 4, usage: storageUsage() }),
    startsByteOffset: 0,
    orderByteOffset: 0,
    cellOfByteOffset: 0,
  });
  const sections = listSections(probe.geometry.cellCount, n, exclusions.list.length);
  const lists = pool.buffer({
    label: "nonbonded/lists",
    byteLength: sections.totalBytes,
    usage: readableStorageUsage(),
  });
  const target: ListTarget = {
    buffer: lists,
    startsByteOffset: sections.cellStarts,
    orderByteOffset: sections.order,
    cellOfByteOffset: sections.cellOf,
  };
  const cellList = prepareCellList(pool, n, boxVectors, searchRadius, target);

  // Only the exclusion sections are static; the cell list's are written by its
  // own kernel every rebuild.
  device.queue.writeBuffer(
    lists,
    sections.exclusionStarts,
    exclusions.starts as unknown as BufferSource,
  );
  if (exclusions.list.length > 0) {
    device.queue.writeBuffer(
      lists,
      sections.exclusionList,
      exclusions.list as unknown as BufferSource,
    );
  }

  const verletParams = pool.buffer({
    label: "verlet/params",
    byteLength: 16,
    usage: uniformUsage(),
  });
  device.queue.writeBuffer(
    verletParams,
    0,
    new Uint32Array([n, 0, 0, 0]) as unknown as BufferSource,
  );
  device.queue.writeBuffer(
    verletParams,
    4,
    new Float32Array([(0.5 * skin) ** 2]) as unknown as BufferSource,
  );
  // 🔴 The list dominates the engine's memory: capacity words an atom, which is
  // 2 kB an atom at a 1 nm cutoff in water. `deviceFitness` has to know, or a
  // preflight promises a system that will not fit.
  const neighbourBytes = Math.max(4, capacity * n * 4);
  const verlet: ResidentVerlet = {
    capacity,
    skin,
    searchRadius,
    neighbours: pool.buffer({
      label: "verlet/neighbours",
      byteLength: neighbourBytes,
      usage: storageUsage(),
    }),
    counts: pool.buffer({
      label: "verlet/counts",
      byteLength: n * 4,
      usage: readableStorageUsage(),
    }),
    // One count per (atom, shell cell), which the prefix sum turns into the
    // offset each cell's fill writes at.
    shellCounts: pool.buffer({
      label: "verlet/shellCounts",
      byteLength: Math.max(4, capacity === 0 ? 4 : NEIGHBOUR_CELLS * n * 4),
      usage: storageUsage(),
    }),
    reference: pool.buffer({
      label: "verlet/reference",
      byteLength: n * 3 * 8,
      usage: storageUsage(),
    }),
    flags: pool.buffer({
      label: "verlet/flags",
      byteLength: 32,
      usage: readableStorageUsage(),
    }),
    params: verletParams,
    byteLength:
      neighbourBytes +
      n * 4 +
      (capacity === 0 ? 0 : NEIGHBOUR_CELLS * n * 4) +
      n * 24 +
      16,
  };
  // `built` starts clear, so the first step builds; `counts` starts at zero so
  // that a run with no list sends every atom to the cell-shell path.
  device.queue.writeBuffer(verlet.flags, 0, new Uint32Array(8) as unknown as BufferSource);
  device.queue.writeBuffer(verlet.counts, 0, new Uint32Array(n) as unknown as BufferSource);

  const particleData = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    particleData[i * 4] = force.charge[i] as number;
    particleData[i * 4 + 1] = force.sigma[i] as number;
    particleData[i * 4 + 2] = force.epsilon[i] as number;
  }
  const particle = pool.buffer({
    label: "nonbonded/particle",
    byteLength: particleData.byteLength,
    usage: storageUsage(),
  });
  device.queue.writeBuffer(particle, 0, particleData as unknown as BufferSource);

  const pairParams = pool.buffer({
    label: "nonbonded/pairParams",
    byteLength: PAIR_PARAMS_BYTES,
    usage: uniformUsage(),
  });
  device.queue.writeBuffer(
    pairParams,
    0,
    packPairParams(
      force,
      sections,
      probe.geometry.divisions,
      box,
      boxVectors,
      alpha,
      verlet,
    ) as unknown as BufferSource,
  );

  // Exceptions: a header vec4 per exception holding the two atom indices as raw
  // bits plus chargeProd, then a parameter vec4 per exception.
  const exceptionData = new Float32Array(Math.max(4, force.exceptionCount * 8));
  const exceptionBits = new Uint32Array(exceptionData.buffer);
  for (let e = 0; e < force.exceptionCount; e++) {
    exceptionBits[e * 4] = force.exceptionP1[e] as number;
    exceptionBits[e * 4 + 1] = force.exceptionP2[e] as number;
    exceptionData[e * 4 + 2] = force.exceptionChargeProd[e] as number;
    const parameters = (force.exceptionCount + e) * 4;
    exceptionData[parameters] = force.exceptionSigma[e] as number;
    exceptionData[parameters + 1] = force.exceptionEpsilon[e] as number;
  }
  const exceptionsBuffer = pool.buffer({
    label: "nonbonded/exceptions",
    byteLength: exceptionData.byteLength,
    usage: storageUsage(),
  });
  device.queue.writeBuffer(
    exceptionsBuffer,
    0,
    exceptionData as unknown as BufferSource,
  );

  const exceptionParams = pool.buffer({
    label: "nonbonded/exceptionParams",
    byteLength: EXCEPTION_PARAMS_BYTES,
    usage: uniformUsage(),
  });
  device.queue.writeBuffer(
    exceptionParams,
    0,
    packExceptionParams(force, box, alpha) as unknown as BufferSource,
  );

  return {
    force,
    particleCount: n,
    box,
    alpha,
    sections,
    lists,
    particle,
    pairParams,
    exceptions: exceptionsBuffer,
    exceptionParams,
    cellList,
    verlet,
    dispersion: force.useDispersionCorrection
      ? dispersionCorrection(
          force,
          (box[0] as number) * (box[1] as number) * (box[2] as number),
        )
      : 0,
  };
}

/**
 * The cells of a neighbour shell: the home cell and the 26 around it.
 *
 * The pair kernel takes one of these per invocation, from the dispatch's y
 * axis, rather than looping over all 27.
 */
const NEIGHBOUR_CELLS = 27;

export interface Accumulators {
  readonly positions: GPUBuffer;
  readonly forces: GPUBuffer;
  readonly energy: GPUBuffer;
}

/** The pair pass, and the exception pass when there are any. */
export interface DirectPasses {
  /**
   * Deciding whether the list has gone stale, then rebuilding it.
   *
   * Encoded every step and almost always doing nothing: four of the five
   * return on the flag. Empty when there is no list.
   */
  readonly rebuild: readonly Pass[];
  /** The neighbour-list path, or null when there is no list. */
  readonly listPairs: Pass | null;
  /**
   * The cell-shell path.
   *
   * With a list this handles only the atoms whose list overflowed, and is a
   * dispatch that returns immediately when none did. Without one it is the
   * whole force.
   */
  readonly pairs: Pass;
  readonly exceptions: Pass | null;
}

export async function planDirect(
  cache: PipelineCache,
  resident: ResidentDirect,
  accumulators: Accumulators,
): Promise<DirectPasses> {
  const { verlet } = resident;
  const atoms = Math.ceil(resident.particleCount / 64);
  // 🔴 One entry at a time, and only the ones the entry point reads. A pipeline
  // created with `layout: "auto"` gets a layout derived from the bindings that
  // entry point *statically uses*, so handing it one it does not touch is
  // rejected outright -- `direct_pairs` never reads `neighbours`, only the
  // count it gates on. The alternative, one shared array for all three entry
  // points, does not survive that.
  //
  // Binding 0 is the *wrapped* positions, not the integrator's. Both endpoints
  // inside the box is what lets the minimum image be a single conditional
  // shift, which is 4x the whole kernel on 3HTB. The `wrap` pass that fills it
  // runs first in the step.
  const wrapped = { binding: 0, resource: { buffer: resident.cellList.wrapped } };
  const forces = { binding: 1, resource: { buffer: accumulators.forces } };
  const energy = { binding: 2, resource: { buffer: accumulators.energy } };
  const particle = { binding: 3, resource: { buffer: resident.particle } };
  const lists = { binding: 4, resource: { buffer: resident.lists } };
  const pairParams = { binding: 5, resource: { buffer: resident.pairParams } };
  const neighbours = { binding: 6, resource: { buffer: verlet.neighbours } };
  const counts = { binding: 7, resource: { buffer: verlet.counts } };
  const flags = { binding: 8, resource: { buffer: verlet.flags } };
  const shellCounts = { binding: 9, resource: { buffer: verlet.shellCounts } };

  const pairs = await cache.pass(
    {
      label: "direct space pairs",
      code: NONBONDED_PAIRS,
      entryPoint: "direct_pairs",
    },
    [wrapped, forces, energy, particle, lists, pairParams, counts],
    // One invocation per (atom, neighbour cell): the y axis carries the 27
    // cells of the shell. The kernel used to loop over them, which left only
    // `particleCount` threads in flight; see the note on `shell` in
    // nonbonded-pairs.wgsl for why that was the cost of the whole step.
    [atoms, NEIGHBOUR_CELLS],
  );

  const listPairs =
    verlet.capacity === 0
      ? null
      : await cache.pass(
          {
            label: "direct space pairs (list)",
            code: NONBONDED_PAIRS,
            entryPoint: "direct_pairs_list",
          },
          [wrapped, forces, energy, particle, lists, pairParams, neighbours, counts],
          // The y axis slices one atom's list; see `direct_pairs_list`.
          [atoms, VERLET_SLICES],
        );

  // Again only what each entry point reads: `reset_rebuild` and `mark_built`
  // touch the flags and nothing else, and an auto layout rejects the rest.
  const positions = { binding: 0, resource: { buffer: accumulators.positions } };
  const referencePositions = { binding: 1, resource: { buffer: verlet.reference } };
  const verletFlags = { binding: 2, resource: { buffer: verlet.flags } };
  const verletParams = { binding: 3, resource: { buffer: verlet.params } };
  const verletPass = (
    entryPoint: string,
    workgroups: number,
    entries: readonly { binding: number; resource: GPUBindingResource }[],
  ): Promise<Pass> =>
    cache.pass(
      { label: `verlet/${entryPoint}`, code: VERLET, entryPoint },
      entries,
      [workgroups],
    );

  const rebuild: readonly Pass[] =
    verlet.capacity === 0
      ? []
      : [
          await verletPass("reset_rebuild", 1, [verletFlags]),
          await verletPass("displacement", atoms, [
            // The *unwrapped* positions: this measures motion, not a pair
            // vector.
            positions,
            referencePositions,
            verletFlags,
            verletParams,
          ]),
          await cache.pass(
            {
              label: "verlet/count",
              code: NONBONDED_PAIRS,
              entryPoint: "count_neighbours",
            },
            [wrapped, lists, pairParams, flags, shellCounts],
            [atoms, NEIGHBOUR_CELLS],
          ),
          await cache.pass(
            {
              label: "verlet/offset",
              code: NONBONDED_PAIRS,
              entryPoint: "offset_neighbours",
            },
            [pairParams, counts, flags, shellCounts],
            [atoms],
          ),
          await cache.pass(
            {
              label: "verlet/fill",
              code: NONBONDED_PAIRS,
              entryPoint: "fill_neighbours",
            },
            [wrapped, lists, pairParams, neighbours, counts, flags, shellCounts],
            [atoms, NEIGHBOUR_CELLS],
          ),
          await verletPass("save_reference", atoms, [
            positions,
            referencePositions,
            verletFlags,
            verletParams,
          ]),
          await verletPass("mark_built", 1, [verletFlags]),
        ];

  if (resident.force.exceptionCount === 0) {
    return { rebuild, listPairs, pairs, exceptions: null };
  }
  const exceptions = await cache.pass(
    {
      label: "direct space exceptions",
      code: NONBONDED_EXCEPTIONS,
      entryPoint: "direct_exceptions",
    },
    [
      { binding: 0, resource: { buffer: accumulators.positions } },
      { binding: 1, resource: { buffer: accumulators.forces } },
      { binding: 2, resource: { buffer: accumulators.energy } },
      { binding: 3, resource: { buffer: resident.particle } },
      { binding: 4, resource: { buffer: resident.exceptions } },
      { binding: 5, resource: { buffer: resident.exceptionParams } },
    ],
    [Math.ceil(resident.force.exceptionCount / 64)],
  );
  return { rebuild, listPairs, pairs, exceptions };
}

/**
 * Write a host-built cell list into the shared buffer.
 *
 * Only for the comparison in src/gpu/validate-cell-list.ts, which needs the two
 * builders' output put through the identical kernel. A run always uses the
 * device's own passes.
 */
export function uploadCellList(
  device: GPUDevice,
  resident: ResidentDirect,
  cellList: CellList,
): void {
  const { sections, lists } = resident;
  device.queue.writeBuffer(
    lists,
    sections.cellStarts,
    cellList.starts as unknown as BufferSource,
  );
  device.queue.writeBuffer(
    lists,
    sections.order,
    cellList.order as unknown as BufferSource,
  );
  const cellOf = new Uint32Array(resident.particleCount);
  for (let cell = 0; cell < cellList.cellCount; cell++) {
    for (
      let slot = cellList.starts[cell] as number;
      slot < (cellList.starts[cell + 1] as number);
      slot++
    ) {
      cellOf[cellList.order[slot] as number] = cell;
    }
  }
  device.queue.writeBuffer(lists, sections.cellOf, cellOf as unknown as BufferSource);
}

/**
 * Evaluate real-space Ewald, Lennard-Jones, the exclusion corrections and the
 * exceptions at one configuration.
 *
 * The pair and exception energies are reported separately, which costs a
 * readback between the two passes. A loop does not do that -- it only needs the
 * total -- but the split is what makes a disagreement in the browser lane
 * readable, so the single-configuration path keeps it.
 */
export async function evaluateDirectSpace(
  device: GPUDevice,
  force: NonbondedForce,
  positionWords: Uint32Array,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  /**
   * A host-built cell list to use instead of the device's own passes, for the
   * comparison that checks the two agree.
   */
  suppliedCellList?: CellList,
): Promise<DirectSpaceResult> {
  void positions;
  const pool = new BufferPool(device);
  try {
    const n = force.particleCount;
    const components = n * 3;
    const accumulators: Accumulators = {
      positions: pool.buffer({
        label: "positions",
        byteLength: positionWords.byteLength,
        usage: storageUsage(),
      }),
      forces: pool.buffer({
        label: "forces",
        byteLength: components * 2 * 4,
        usage: readableStorageUsage(),
      }),
      energy: pool.buffer({
        label: "energy",
        byteLength: 8,
        usage: readableStorageUsage(),
      }),
    };
    device.queue.writeBuffer(
      accumulators.positions,
      0,
      positionWords as unknown as BufferSource,
    );
    device.queue.writeBuffer(accumulators.forces, 0, new Uint32Array(components * 2));
    device.queue.writeBuffer(accumulators.energy, 0, new Uint32Array(2));

    const cache = pipelineCacheFor(device);
    // `0` builds no list: this is the single-configuration path the
    // differential tests use, and it takes the cell-shell route so the
    // reference is compared against `direct_pairs` directly.
    const resident = prepareDirect(pool, force, boxVectors, alpha, 0);
    // The fold runs either way: the pair kernel differences its output whether
    // or not this device built the list.
    await submitPasses(device, "wrap", [
      await planWrapPositions(cache, resident.cellList, accumulators.positions),
    ]);
    if (suppliedCellList === undefined) {
      const passes = await planCellList(cache, resident.cellList, accumulators.positions);
      await submitPasses(device, "cell list", passes);
    } else {
      uploadCellList(device, resident, suppliedCellList);
    }

    const { pairs, exceptions } = await planDirect(cache, resident, accumulators);
    await submitPasses(device, "direct space pairs", [pairs]);
    const afterPairs = new Uint32Array(await readBuffer(device, accumulators.energy, 8));
    const pairEnergy = fromFixed(
      fromWords(afterPairs[0] as number, (afterPairs[1] as number) | 0),
    );
    if (exceptions !== null) {
      await submitPasses(device, "direct space exceptions", [exceptions]);
    }

    const energyWords = new Uint32Array(
      await readBuffer(device, accumulators.energy, 8),
    );
    const forceWords = new Uint32Array(
      await readBuffer(device, accumulators.forces, components * 2 * 4),
    );
    const total = fromFixed(
      fromWords(energyWords[0] as number, (energyWords[1] as number) | 0),
    );
    return {
      pairEnergy,
      exceptionEnergy: total - pairEnergy,
      dispersion: resident.dispersion,
      energy: total + resident.dispersion,
      forces: unpackFixed(forceWords, components),
    };
  } finally {
    pool.destroy();
  }
}
