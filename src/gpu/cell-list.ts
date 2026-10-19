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

import { dispatch, readBuffer } from "./compute.js";
import { CELL_LIST } from "./kernels.js";
import { BufferPool } from "./allocator.js";
import { fixedWords } from "../reference/fixed-point.js";
import { CellListTooCoarse, MINIMUM_CELLS_PER_AXIS } from "./nonbonded.js";
import type { Vec3 } from "../engine/system.js";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

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

/** The device buffers the list lives in. */
export interface CellListBuffers {
  readonly cellOf: GPUBuffer;
  readonly counts: GPUBuffer;
  readonly starts: GPUBuffer;
  /** Whatever the atomic scatter produced. Scratch; nothing downstream reads it. */
  readonly scattered: GPUBuffer;
  /** The same atoms in ascending index order within each cell. This is the list. */
  readonly order: GPUBuffer;
  readonly params: GPUBuffer;
  readonly geometry: CellGeometry;
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
): CellListBuffers {
  const geometry = cellGeometry(box, cutoff);
  const named = (label: string, byteLength: number, usage = STORAGE): GPUBuffer =>
    pool.buffer({ label, byteLength, usage });
  const params = named("cell/params", 128, UNIFORM);
  pool.device.queue.writeBuffer(
    params,
    0,
    packParams(particleCount, geometry, box) as unknown as BufferSource,
  );
  return {
    cellOf: named("cell/cellOf", particleCount * 4),
    counts: named("cell/counts", geometry.cellCount * 4),
    starts: named("cell/starts", (geometry.cellCount + 1) * 4),
    scattered: named("cell/scattered", particleCount * 4),
    order: named("cell/order", particleCount * 4),
    params,
    geometry,
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
export async function buildCellListOnDevice(
  pool: BufferPool,
  buffers: CellListBuffers,
  positions: GPUBuffer,
  particleCount: number,
): Promise<void> {
  const cells = Math.ceil(buffers.geometry.cellCount / 64);
  const atoms = Math.ceil(particleCount / 64);
  const run = async (
    entryPoint: string,
    workgroups: number,
    entries: readonly { binding: number; resource: GPUBindingResource }[],
  ): Promise<void> =>
    dispatch(pool.device, {
      label: `cell/${entryPoint}`,
      code: CELL_LIST,
      entryPoint,
      bindings: entries,
      workgroups: [workgroups],
    });

  const paramsBinding = { binding: 5, resource: { buffer: buffers.params } };
  const positionsBinding = { binding: 0, resource: { buffer: positions } };
  const cellOfBinding = { binding: 1, resource: { buffer: buffers.cellOf } };
  const countsBinding = { binding: 2, resource: { buffer: buffers.counts } };
  const startsBinding = { binding: 3, resource: { buffer: buffers.starts } };
  const scatteredBinding = { binding: 4, resource: { buffer: buffers.scattered } };
  const sortedBinding = { binding: 6, resource: { buffer: buffers.order } };

  await run("clear", cells, [countsBinding, paramsBinding]);
  await run("assign", atoms, [
    positionsBinding,
    cellOfBinding,
    countsBinding,
    paramsBinding,
  ]);
  await run("scan", 1, [countsBinding, startsBinding, paramsBinding]);
  await run("scatter", atoms, [
    cellOfBinding,
    countsBinding,
    startsBinding,
    scatteredBinding,
    paramsBinding,
  ]);
  await run("sort_cells", atoms, [
    cellOfBinding,
    startsBinding,
    scatteredBinding,
    paramsBinding,
    sortedBinding,
  ]);
}

/** Read the built list back, for validation and for the host force path. */
export async function readCellList(
  pool: BufferPool,
  buffers: CellListBuffers,
  particleCount: number,
): Promise<{ starts: Uint32Array; order: Uint32Array; cellOf: Uint32Array }> {
  const [starts, order, cellOf] = await Promise.all([
    readBuffer(pool.device, buffers.starts, (buffers.geometry.cellCount + 1) * 4),
    readBuffer(pool.device, buffers.order, particleCount * 4),
    readBuffer(pool.device, buffers.cellOf, particleCount * 4),
  ]);
  return {
    starts: new Uint32Array(starts),
    order: new Uint32Array(order),
    cellOf: new Uint32Array(cellOf),
  };
}
