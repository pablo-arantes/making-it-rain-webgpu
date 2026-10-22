/**
 * Host side of the device cell list.
 *
 * The list is rebuilt from whatever is in the position buffer, with no readback
 * and no host arithmetic on coordinates -- which is the whole point. A
 * host-built list would mean copying every position back each step, and a
 * readback drains the pipeline, so it would cost more than the forces it exists
 * to make cheap.
 *
 * The geometry is still decided here, because it depends only on the box and the
 * cutoff: how many cells along each axis, and how wide one is. Both are
 * converted to fixed point on the host and compared as integers on the device,
 * so an atom near a cell boundary lands where it belongs even at the far edge
 * of a 6.5 nm box, where an f32 coordinate has a 4.8e-7 nm ulp.
 */

import { pipelineCacheFor, readBuffer } from "./compute.js";
import { submitPasses, type Pass, type PipelineCache } from "./pipeline.js";
import { CELL_LIST } from "./kernels.js";
import { BufferPool } from "./allocator.js";
import { readableStorageUsage, storageUsage, uniformUsage } from "./usage.js";
import { fixedWords } from "../reference/fixed-point.js";
import {
  CellListTooCoarse,
  MINIMUM_CELLS_PER_AXIS,
  writeInverseBox,
  writeOffDiagonalBox,
} from "./nonbonded.js";
import type { Vec3 } from "../engine/system.js";


/** The grid a box and a cutoff imply. Independent of the coordinates. */
export interface CellGeometry {
  readonly divisions: Vec3;
  readonly cellCount: number;
  /** nm, one cell's extent along each axis. */
  readonly extent: Vec3;
}

export class BoxTooSmallForCells extends Error {
  constructor(readonly extent: number, readonly cutoff: number) {
    super(
      `the box has shrunk until a cell is ${extent.toFixed(4)} nm across, below ` +
        `the ${cutoff} nm cutoff. The 27-cell scan only finds every pair inside ` +
        "the cutoff while a cell is at least that wide, so the forces would be " +
        "silently missing pairs, and stopping is the only safe answer.\n\n" +
        "The divisions are fixed for a run's life -- the cell count sizes the " +
        "buffer the pair kernel indexes -- but raising them would not help " +
        "either: fewer, larger cells is what a smaller box needs, and below " +
        `${MINIMUM_CELLS_PER_AXIS} per axis the 27-cell scan visits the same ` +
        "cell twice. A box narrower than three cutoffs cannot use this " +
        "structure at all.\n\n" +
        "That is usually the box's fault rather than the barostat's. 1L2Y at " +
        "the notebook's default padding is 3.08 nm wide against a 1 nm cutoff " +
        "-- 3.08 cutoffs, so 2.6% of compression exhausts it -- and its " +
        "minimum solute-to-own-image distance is already 0.958 nm, below the " +
        "cutoff (docs/NOTEBOOK_PARITY.md finding 4). Either solvate with more " +
        "padding or shorten the cutoff; an all-pairs fallback for tight boxes " +
        "is not built yet.",
    );
    this.name = "BoxTooSmallForCells";
  }
}

/**
 * How much the box can shrink before the cell list stops being usable.
 *
 * Returned as a fraction: 0.026 means 2.6% of compression exhausts it. A caller
 * running NPT wants to know this *before* starting, because the alternative is
 * finding out partway through a trajectory.
 */
export function cellHeadroom(box: Vec3, cutoff: number): number {
  const geometry = cellGeometry(box, cutoff);
  let worst = Number.POSITIVE_INFINITY;
  for (const extent of geometry.extent) {
    worst = Math.min(worst, ((extent as number) - cutoff) / (extent as number));
  }
  return worst;
}

/**
 * The grid for a box whose divisions are already fixed.
 *
 * A barostat changes the box, and `floor(box / cutoff)` would then change the
 * *number* of cells -- which decides the size of the buffer the pair kernel
 * indexes into, and so cannot change during a run. The divisions stay as the
 * starting box set them and the extents move with the volume, which stays
 * correct as long as an extent does not fall below the cutoff. The starting
 * extent is `box / floor(box / cutoff)`, so the headroom is whatever the box
 * happens to leave: 8.7% for 3HTB's 6.52 nm box at a 1 nm cutoff. Well beyond
 * anything an equilibrated barostat does, and checked rather than assumed.
 */
export function geometryForDivisions(
  box: Vec3,
  divisions: Vec3,
  cutoff: number,
): CellGeometry {
  const extent = [0, 1, 2].map(
    (axis) => (box[axis] as number) / (divisions[axis] as number),
  ) as unknown as Vec3;
  for (const value of extent) {
    if ((value as number) < cutoff) {
      throw new BoxTooSmallForCells(value as number, cutoff);
    }
  }
  return {
    divisions,
    cellCount:
      (divisions[0] as number) * (divisions[1] as number) * (divisions[2] as number),
    extent,
  };
}

export function cellGeometry(box: Vec3, cutoff: number): CellGeometry {
  const divisions: Vec3 = [
    Math.max(1, Math.floor((box[0] as number) / cutoff)),
    Math.max(1, Math.floor((box[1] as number) / cutoff)),
    Math.max(1, Math.floor((box[2] as number) / cutoff)),
  ];
  if (divisions.some((n) => n < MINIMUM_CELLS_PER_AXIS)) {
    throw new CellListTooCoarse(divisions, box, cutoff);
  }
  return {
    divisions,
    cellCount:
      (divisions[0] as number) * (divisions[1] as number) * (divisions[2] as number),
    extent: [
      (box[0] as number) / (divisions[0] as number),
      (box[1] as number) / (divisions[1] as number),
      (box[2] as number) / (divisions[2] as number),
    ],
  };
}

/**
 * Where the list's three arrays live inside a caller's buffer.
 *
 * The pair kernel reads the cell starts, the sorted order, each atom's cell and
 * the exclusion lists through a *single* storage binding, because splitting
 * them would put it over the eight storage buffers WebGPU guarantees per stage.
 * So the cell-list kernel writes into sub-ranges of that same buffer rather
 * than into buffers of its own, and nothing is copied between them.
 *
 * Offsets are in bytes and must be multiples of 256, which is WebGPU's default
 * `minStorageBufferOffsetAlignment`; `alignedBytes` is what the caller lays its
 * sections out with.
 */
export interface ListTarget {
  readonly buffer: GPUBuffer;
  readonly startsByteOffset: number;
  readonly orderByteOffset: number;
  readonly cellOfByteOffset: number;
}

/** Round a byte count up to a bindable offset. */
export const alignedBytes = (bytes: number): number => Math.ceil(bytes / 256) * 256;

/** The device buffers the list lives in. */
export interface CellListBuffers {
  readonly counts: GPUBuffer;
  /** Whatever the atomic scatter produced. Scratch; nothing downstream reads it. */
  readonly scattered: GPUBuffer;
  /**
   * Every coordinate folded into [0, box), written by the `wrap` pass.
   *
   * This is what the pair kernel differences, rather than the integrator's
   * positions. Both endpoints inside the box is the precondition that lets its
   * minimum image be a single conditional shift; see `nearest_image_box`.
   */
  readonly wrapped: GPUBuffer;
  readonly params: GPUBuffer;
  readonly geometry: CellGeometry;
  readonly target: ListTarget;
  readonly particleCount: number;
}

/**
 * Pack `CellParams`.
 *
 * 128 bytes. The two fixed-point arrays have a 16-byte stride because that is
 * what WGSL gives an array element in the uniform address space, and the
 * shader declares them as `vec4` so the padding is visible on both sides.
 */
function packParams(
  particleCount: number,
  geometry: CellGeometry,
  box: Vec3,
  boxVectors?: readonly [Vec3, Vec3, Vec3],
): Uint32Array {
  const words = new Uint32Array(56);
  words[0] = particleCount;
  words[1] = geometry.cellCount;
  for (let axis = 0; axis < 3; axis++) {
    words[4 + axis] = geometry.divisions[axis] as number;
    const boxWords = fixedWords(box[axis] as number);
    words[8 + axis * 4] = boxWords.lo;
    words[8 + axis * 4 + 1] = boxWords.hi;
    const extentWords = fixedWords(geometry.extent[axis] as number);
    words[20 + axis * 4] = extentWords.lo;
    words[20 + axis * 4 + 1] = extentWords.hi;
  }
  // The off-diagonals and the inverse box, appended. A rectangular cell leaves
  // the off-diagonals zero and the inverse diagonal, which is what `skewed()`
  // in the kernel reads to keep the exact integer paths.
  const vectors = boxVectors ?? diagonalVectors(box);
  writeOffDiagonalBox(words, 32, vectors);
  writeInverseBox(new Float32Array(words.buffer), 44, vectors);
  return words;
}

/** Box vectors from three edge lengths, for a caller that only has those. */
const diagonalVectors = (box: Vec3): readonly [Vec3, Vec3, Vec3] => [
  [box[0] as number, 0, 0],
  [0, box[1] as number, 0],
  [0, 0, box[2] as number],
];

/**
 * Rewrite the geometry uniform for a new box, keeping the divisions.
 *
 * Throws `BoxTooSmallForCells` rather than quietly producing a list that misses
 * pairs, which is what a cell narrower than the cutoff would do.
 */
export function updateCellBox(
  pool: BufferPool,
  buffers: CellListBuffers,
  box: Vec3,
  cutoff: number,
): CellGeometry {
  const geometry = geometryForDivisions(box, buffers.geometry.divisions, cutoff);
  pool.device.queue.writeBuffer(
    buffers.params,
    0,
    packParams(buffers.particleCount, geometry, box) as unknown as BufferSource,
  );
  return geometry;
}

/** Create (or find) the list's buffers and upload the geometry. */
export function prepareCellList(
  pool: BufferPool,
  particleCount: number,
  box: Vec3,
  cutoff: number,
  target: ListTarget,
): CellListBuffers {
  const geometry = cellGeometry(box, cutoff);
  const named = (label: string, byteLength: number, usage = readableStorageUsage()): GPUBuffer =>
    pool.buffer({ label, byteLength, usage });
  const params = named("cell/params", 224, uniformUsage());
  pool.device.queue.writeBuffer(
    params,
    0,
    packParams(particleCount, geometry, box) as unknown as BufferSource,
  );
  return {
    counts: named("cell/counts", geometry.cellCount * 4),
    scattered: named("cell/scattered", particleCount * 4),
    // Two words per component, like the positions it folds.
    wrapped: named("cell/wrapped", particleCount * 3 * 8, storageUsage()),
    params,
    geometry,
    target,
    particleCount,
  };
}

/**
 * The fold on its own.
 *
 * 🔴 Every path that encodes a pair pass has to encode this first, and it is
 * deliberately *not* part of `planCellList`: the path that uploads a host-built
 * list never calls that at all, and would leave the kernel differencing a
 * buffer of zeros -- a wrong answer, not an error.
 */
export async function planWrapPositions(
  cache: PipelineCache,
  buffers: CellListBuffers,
  positions: GPUBuffer,
): Promise<Pass> {
  return cache.pass(
    { label: "cell/wrap", code: CELL_LIST, entryPoint: "wrap" },
    [
      { binding: 0, resource: { buffer: positions } },
      { binding: 5, resource: { buffer: buffers.params } },
      { binding: 7, resource: { buffer: buffers.wrapped } },
    ],
    [Math.ceil(buffers.particleCount / 64)],
  );
}

/**
 * Rebuild the list from the positions in `positions`.
 *
 * Five dispatches, no readback. `clear` is cheap and run every time rather than
 * relying on `scan` having left the counts at zero, so a build is correct
 * whatever ran before it. `sort_cells` is the fifth, and it is not optional --
 * though not for the reason it was written for: the pair kernel now adds every
 * pair in fixed point, so the order within a cell cannot change a force. It is
 * what makes the device's list comparable to the host builder's, and what gives
 * the neighbour list built from it a defined order. The kernel's own header has
 * the measurement and the history.
 *
 * 🔴 It does **not** include `planWrapPositions`, and must not: the fold has to
 * run every step even where the list is rebuilt less often, because the pair
 * kernel differences the fold's output. Callers encode both.
 */
export async function planCellList(
  cache: PipelineCache,
  buffers: CellListBuffers,
  positions: GPUBuffer,
): Promise<Pass[]> {
  const { geometry, target, particleCount } = buffers;
  const cells = Math.ceil(geometry.cellCount / 64);
  const atoms = Math.ceil(particleCount / 64);

  // Sub-ranges of the caller's buffer, bound as if they were whole arrays, so
  // each entry point keeps indexing from zero.
  const range = (offset: number, byteLength: number): GPUBindingResource => ({
    buffer: target.buffer,
    offset,
    size: byteLength,
  });
  const paramsBinding = { binding: 5, resource: { buffer: buffers.params } };
  const positionsBinding = { binding: 0, resource: { buffer: positions } };
  const cellOfBinding = {
    binding: 1,
    resource: range(target.cellOfByteOffset, particleCount * 4),
  };
  const countsBinding = { binding: 2, resource: { buffer: buffers.counts } };
  const startsBinding = {
    binding: 3,
    resource: range(target.startsByteOffset, (geometry.cellCount + 1) * 4),
  };
  const scatteredBinding = { binding: 4, resource: { buffer: buffers.scattered } };
  const sortedBinding = {
    binding: 6,
    resource: range(target.orderByteOffset, particleCount * 4),
  };

  const plan = (
    entryPoint: string,
    workgroups: number,
    entries: readonly { binding: number; resource: GPUBindingResource }[],
  ): Promise<Pass> =>
    cache.pass(
      { label: `cell/${entryPoint}`, code: CELL_LIST, entryPoint },
      entries,
      [workgroups],
    );

  return Promise.all([
    plan("clear", cells, [countsBinding, paramsBinding]),
    // 🔴 `assign` reads the *wrapped* positions now, for a skewed cell's
    // fractional bucketing, so binding 7 has to be here: an auto layout is
    // derived from what the entry point statically uses and rejects a bind
    // group that is missing one.
    plan("assign", atoms, [
      positionsBinding,
      cellOfBinding,
      countsBinding,
      paramsBinding,
      { binding: 7, resource: { buffer: buffers.wrapped } },
    ]),
    plan("scan", 1, [countsBinding, startsBinding, paramsBinding]),
    plan("scatter", atoms, [
      cellOfBinding,
      countsBinding,
      startsBinding,
      scatteredBinding,
      paramsBinding,
    ]),
    plan("sort_cells", atoms, [
      cellOfBinding,
      startsBinding,
      scatteredBinding,
      paramsBinding,
      sortedBinding,
    ]),
  ]);
}

/** Build the list once, for a single-configuration evaluation. */
export async function buildCellListOnDevice(
  pool: BufferPool,
  buffers: CellListBuffers,
  positions: GPUBuffer,
): Promise<void> {
  const cache = pipelineCacheFor(pool.device);
  await submitPasses(pool.device, "cell list", [
    await planWrapPositions(cache, buffers, positions),
    ...(await planCellList(cache, buffers, positions)),
  ]);
}

/** Read the built list back, for validation and for the host force path. */
export async function readCellList(
  pool: BufferPool,
  buffers: CellListBuffers,
): Promise<{ starts: Uint32Array; order: Uint32Array; cellOf: Uint32Array }> {
  const { target, geometry, particleCount } = buffers;
  const words = new Uint32Array(
    await readBuffer(pool.device, target.buffer, target.buffer.size),
  );
  const slice = (byteOffset: number, count: number): Uint32Array =>
    words.slice(byteOffset / 4, byteOffset / 4 + count);
  return {
    starts: slice(target.startsByteOffset, geometry.cellCount + 1),
    order: slice(target.orderByteOffset, particleCount),
    cellOf: slice(target.cellOfByteOffset, particleCount),
  };
}
