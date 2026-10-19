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
import { readableStorageUsage, uniformUsage } from "./usage.js";
import { fixedWords } from "../reference/fixed-point.js";
import { CellListTooCoarse, MINIMUM_CELLS_PER_AXIS } from "./nonbonded.js";
import type { Vec3 } from "../engine/system.js";


/** The grid a box and a cutoff imply. Independent of the coordinates. */
export interface CellGeometry {
  readonly divisions: Vec3;
  readonly cellCount: number;
  /** nm, one cell's extent along each axis. */
  readonly extent: Vec3;
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
): Uint32Array {
  const words = new Uint32Array(32);
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
  return words;
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
  const params = named("cell/params", 128, uniformUsage());
  pool.device.queue.writeBuffer(
    params,
    0,
    packParams(particleCount, geometry, box) as unknown as BufferSource,
  );
  return {
    counts: named("cell/counts", geometry.cellCount * 4),
    scattered: named("cell/scattered", particleCount * 4),
    params,
    geometry,
    target,
    particleCount,
  };
}

/**
 * Rebuild the list from the positions in `positions`.
 *
 * Five dispatches, no readback. `clear` is cheap and run every time rather than
 * relying on `scan` having left the counts at zero, so a build is correct
 * whatever ran before it. `sort_cells` is the fifth, and it is not optional:
 * the pair kernel sums one atom's neighbours in an f32 register, so the order
 * within a cell changes the answer -- measured at 1.89e-3 kJ/mol/nm, which is
 * small and is still run-to-run noise in a trajectory.
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
    plan("assign", atoms, [
      positionsBinding,
      cellOfBinding,
      countsBinding,
      paramsBinding,
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
  const passes = await planCellList(pipelineCacheFor(pool.device), buffers, positions);
  await submitPasses(pool.device, "cell list", passes);
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
