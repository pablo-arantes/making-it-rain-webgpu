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
  planCellList,
  prepareCellList,
  type CellListBuffers,
  type ListTarget,
} from "./cell-list.js";
import { NONBONDED_EXCEPTIONS, NONBONDED_PAIRS } from "./kernels.js";
import { unpackFixed } from "./forces.js";
import { fixedWords, fromFixed, fromWords } from "../reference/fixed-point.js";
import {
  dispersionCorrection,
  rectangularBox,
} from "../reference/forces/nonbonded.js";
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
  box: Vec3,
  cutoff: number,
): CellList {
  const particleCount = positions.length / 3;
  const cells: Vec3 = [
    Math.max(1, Math.floor((box[0] as number) / cutoff)),
    Math.max(1, Math.floor((box[1] as number) / cutoff)),
    Math.max(1, Math.floor((box[2] as number) / cutoff)),
  ];
  if (cells.some((n) => n < MINIMUM_CELLS_PER_AXIS)) {
    throw new CellListTooCoarse(cells, box, cutoff);
  }
  const cellCount = (cells[0] as number) * (cells[1] as number) * (cells[2] as number);
  const cellOf = new Int32Array(particleCount);
  const starts = new Uint32Array(cellCount + 1);
  const index = (value: number, extent: number, divisions: number): number => {
    const fraction = value / extent - Math.floor(value / extent);
    const i = Math.floor(fraction * divisions);
    return i === divisions ? divisions - 1 : i;
  };
  for (let i = 0; i < particleCount; i++) {
    const cx = index(positions[i * 3] as number, box[0] as number, cells[0] as number);
    const cy = index(positions[i * 3 + 1] as number, box[1] as number, cells[1] as number);
    const cz = index(positions[i * 3 + 2] as number, box[2] as number, cells[2] as number);
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
 * 160 bytes. Five u32 word offsets fill 0..19; `cells` takes its 16-byte
 * boundary at 32 and the cutoff follows at 44; the f32 box takes the next one
 * at 48 with alpha at 60; then the fixed-point box at 64 and half the box at
 * 112, three `vec4<u32>` each at WGSL's 16-byte array stride.
 */
export const PAIR_PARAMS_BYTES = 160;

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
  alpha: number,
): Uint8Array {
  const bytes = new ArrayBuffer(PAIR_PARAMS_BYTES);
  const u32 = new Uint32Array(bytes);
  const f32 = new Float32Array(bytes);
  u32[0] = force.particleCount;
  u32[1] = sections.order / 4;
  u32[2] = sections.exclusionStarts / 4;
  u32[3] = sections.exclusionList / 4;
  u32[4] = sections.cellOf / 4;
  u32[8] = divisions[0] as number;
  u32[9] = divisions[1] as number;
  u32[10] = divisions[2] as number;
  f32[11] = force.cutoff;
  f32[12] = box[0] as number;
  f32[13] = box[1] as number;
  f32[14] = box[2] as number;
  f32[15] = alpha;
  writeFixedBox(u32, 16, box);
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
}

export function prepareDirect(
  pool: BufferPool,
  force: NonbondedForce,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
): ResidentDirect {
  const device = pool.device;
  const box = rectangularBox(boxVectors);
  const n = force.particleCount;
  const exclusions = buildExclusionLists(force);

  // The grid has to be known before the sections can be laid out.
  const probe = prepareCellList(pool, n, box, force.cutoff, {
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
  const cellList = prepareCellList(pool, n, box, force.cutoff, target);

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
      alpha,
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
    dispersion: force.useDispersionCorrection
      ? dispersionCorrection(
          force,
          (box[0] as number) * (box[1] as number) * (box[2] as number),
        )
      : 0,
  };
}

export interface Accumulators {
  readonly positions: GPUBuffer;
  readonly forces: GPUBuffer;
  readonly energy: GPUBuffer;
}

/** The pair pass, and the exception pass when there are any. */
export async function planDirect(
  cache: PipelineCache,
  resident: ResidentDirect,
  accumulators: Accumulators,
): Promise<{ pairs: Pass; exceptions: Pass | null }> {
  const pairs = await cache.pass(
    {
      label: "direct space pairs",
      code: NONBONDED_PAIRS,
      entryPoint: "direct_pairs",
    },
    [
      { binding: 0, resource: { buffer: accumulators.positions } },
      { binding: 1, resource: { buffer: accumulators.forces } },
      { binding: 2, resource: { buffer: accumulators.energy } },
      { binding: 3, resource: { buffer: resident.particle } },
      { binding: 4, resource: { buffer: resident.lists } },
      { binding: 5, resource: { buffer: resident.pairParams } },
    ],
    [Math.ceil(resident.particleCount / 64)],
  );
  if (resident.force.exceptionCount === 0) return { pairs, exceptions: null };
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
  return { pairs, exceptions };
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
    const resident = prepareDirect(pool, force, boxVectors, alpha);
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
