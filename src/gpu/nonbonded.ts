/**
 * Host side of the real-space nonbonded kernels.
 *
 * Builds the cell list and the per-atom exclusion lists, runs the two
 * dispatches into one shared force and energy accumulator, and adds the
 * analytic dispersion correction, which depends on the volume but on no
 * coordinate and so is a host-side constant.
 *
 * The cell list is built on the host here. That is a Phase 1 decision, not a
 * final one: a running simulation needs it rebuilt on the device every few
 * steps, which is a counting sort in WGSL. Correctness of the force kernels
 * comes first, and they do not care who sorted the atoms.
 */

import { dispatch, readBuffer, storageBuffer, uploadBuffer } from "./compute.js";
import { NONBONDED_EXCEPTIONS, NONBONDED_PAIRS } from "./kernels.js";
import { unpackFixed } from "./forces.js";
import { fromFixed, fromWords } from "../reference/fixed-point.js";
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

/** Zero a two-word fixed-point accumulator of `slots` slots. */
function zeroAccumulator(device: GPUDevice, buffer: GPUBuffer, slots: number): void {
  device.queue.writeBuffer(buffer, 0, new Uint32Array(slots * 2));
}

/**
 * Evaluate real-space Ewald, Lennard-Jones, the exclusion corrections and the
 * exceptions on the device, in one shared accumulator.
 */
export async function evaluateDirectSpace(
  device: GPUDevice,
  force: NonbondedForce,
  positionWords: Uint32Array,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  /**
   * A cell list to use instead of building one here.
   *
   * The step loop builds its list on the device and never brings the positions
   * back; this is also what lets the device-built list be compared against the
   * host's by the only measure that matters -- whether the forces come out the
   * same. See src/gpu/validate-cell-list.ts.
   */
  suppliedCellList?: CellList,
): Promise<DirectSpaceResult> {
  const box = rectangularBox(boxVectors);
  const n = force.particleCount;
  const cellList = suppliedCellList ?? buildCellList(positions, box, force.cutoff);
  const exclusions = buildExclusionLists(force);

  // One `lists` buffer so the kernel stays inside the eight storage bindings
  // WebGPU guarantees: cell starts, then the sorted order, then the exclusion
  // starts, then the flat exclusion list.
  const sortedOffset = cellList.cellCount + 1;
  const exclusionStartOffset = sortedOffset + n;
  const exclusionListOffset = exclusionStartOffset + n + 1;
  const cellOfOffset = exclusionListOffset + exclusions.list.length;
  const lists = new Uint32Array(cellOfOffset + n);
  lists.set(cellList.starts, 0);
  lists.set(cellList.order, sortedOffset);
  lists.set(exclusions.starts, exclusionStartOffset);
  lists.set(exclusions.list, exclusionListOffset);
  // Each atom's cell, so the kernel reads its home cell instead of recomputing
  // it in f32 and possibly disagreeing with the list it is indexing into.
  for (let cell = 0; cell < cellList.cellCount; cell++) {
    for (
      let slot = cellList.starts[cell] as number;
      slot < (cellList.starts[cell + 1] as number);
      slot++
    ) {
      lists[cellOfOffset + (cellList.order[slot] as number)] = cell;
    }
  }

  const particle = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    particle[i * 4] = force.charge[i] as number;
    particle[i * 4 + 1] = force.sigma[i] as number;
    particle[i * 4 + 2] = force.epsilon[i] as number;
  }

  // Exceptions: a header vec4 per exception holding the two atom indices as raw
  // bits plus chargeProd, then a parameter vec4 per exception.
  const exceptionData = new Float32Array(force.exceptionCount * 8);
  const exceptionBits = new Uint32Array(exceptionData.buffer);
  for (let e = 0; e < force.exceptionCount; e++) {
    exceptionBits[e * 4] = force.exceptionP1[e] as number;
    exceptionBits[e * 4 + 1] = force.exceptionP2[e] as number;
    exceptionData[e * 4 + 2] = force.exceptionChargeProd[e] as number;
    const parameters = (force.exceptionCount + e) * 4;
    exceptionData[parameters] = force.exceptionSigma[e] as number;
    exceptionData[parameters + 1] = force.exceptionEpsilon[e] as number;
  }

  const positionBuffer = uploadBuffer(device, {
    label: "positions",
    data: positionWords,
    usage: GPUBufferUsage.STORAGE,
  });
  const forceBuffer = storageBuffer(device, "forces", n * 3 * 2 * 4);
  zeroAccumulator(device, forceBuffer, n * 3);
  const energyBuffer = storageBuffer(device, "energy", 8);
  zeroAccumulator(device, energyBuffer, 1);
  const particleBuffer = uploadBuffer(device, {
    label: "particle parameters",
    data: particle,
    usage: GPUBufferUsage.STORAGE,
  });
  const listBuffer = uploadBuffer(device, {
    label: "cell and exclusion lists",
    data: lists,
    usage: GPUBufferUsage.STORAGE,
  });

  // struct Params { u32 x4; vec3<u32> cells; f32 cutoff; vec3<f32> box; f32 alpha }
  // vec3 has 16-byte alignment, so cells lands at 16 and box at 32; 48 bytes.
  // 64 bytes: five u32 offsets, then `cells` on its 16-byte boundary, the
  // cutoff, the box on the next one, and alpha.
  const pairParams = new ArrayBuffer(64);
  const pairU32 = new Uint32Array(pairParams);
  const pairF32 = new Float32Array(pairParams);
  pairU32[0] = n;
  pairU32[1] = sortedOffset;
  pairU32[2] = exclusionStartOffset;
  pairU32[3] = exclusionListOffset;
  pairU32[4] = cellOfOffset;
  pairU32[8] = cellList.cells[0] as number;
  pairU32[9] = cellList.cells[1] as number;
  pairU32[10] = cellList.cells[2] as number;
  pairF32[11] = force.cutoff;
  pairF32[12] = box[0] as number;
  pairF32[13] = box[1] as number;
  pairF32[14] = box[2] as number;
  pairF32[15] = alpha;
  const pairParamBuffer = uploadBuffer(device, {
    label: "pair params",
    data: new Uint8Array(pairParams),
    usage: GPUBufferUsage.UNIFORM,
  });

  const exceptionBuffer = uploadBuffer(device, {
    label: "exception data",
    // An empty buffer is invalid, so a system with no exceptions still gets one
    // vec4 that no invocation reads.
    data: force.exceptionCount === 0 ? new Float32Array(4) : exceptionData,
    usage: GPUBufferUsage.STORAGE,
  });
  // struct Params { u32 count; f32 alpha; vec3<f32> box; u32 offset } -> 32 bytes
  const exceptionParams = new ArrayBuffer(32);
  const exceptionU32 = new Uint32Array(exceptionParams);
  const exceptionF32 = new Float32Array(exceptionParams);
  exceptionU32[0] = force.exceptionCount;
  exceptionF32[1] = alpha;
  exceptionF32[4] = box[0] as number;
  exceptionF32[5] = box[1] as number;
  exceptionF32[6] = box[2] as number;
  exceptionU32[7] = force.exceptionCount;
  const exceptionParamBuffer = uploadBuffer(device, {
    label: "exception params",
    data: new Uint8Array(exceptionParams),
    usage: GPUBufferUsage.UNIFORM,
  });

  const buffers = [
    positionBuffer, forceBuffer, energyBuffer, particleBuffer, listBuffer,
    pairParamBuffer, exceptionBuffer, exceptionParamBuffer,
  ];
  try {
    await dispatch(device, {
      label: "direct space pairs",
      code: NONBONDED_PAIRS,
      entryPoint: "direct_pairs",
      bindings: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: forceBuffer } },
        { binding: 2, resource: { buffer: energyBuffer } },
        { binding: 3, resource: { buffer: particleBuffer } },
        { binding: 4, resource: { buffer: listBuffer } },
        { binding: 5, resource: { buffer: pairParamBuffer } },
      ],
      workgroups: [Math.ceil(n / 64)],
    });
    const afterPairs = new Uint32Array(await readBuffer(device, energyBuffer, 8));
    const pairEnergy = fromFixed(
      fromWords(afterPairs[0] as number, (afterPairs[1] as number) | 0),
    );

    if (force.exceptionCount > 0) {
      await dispatch(device, {
        label: "direct space exceptions",
        code: NONBONDED_EXCEPTIONS,
        entryPoint: "direct_exceptions",
        bindings: [
          { binding: 0, resource: { buffer: positionBuffer } },
          { binding: 1, resource: { buffer: forceBuffer } },
          { binding: 2, resource: { buffer: energyBuffer } },
          { binding: 3, resource: { buffer: particleBuffer } },
          { binding: 4, resource: { buffer: exceptionBuffer } },
          { binding: 5, resource: { buffer: exceptionParamBuffer } },
        ],
        workgroups: [Math.ceil(force.exceptionCount / 64)],
      });
    }

    const energyWords = new Uint32Array(await readBuffer(device, energyBuffer, 8));
    const forceWords = new Uint32Array(
      await readBuffer(device, forceBuffer, n * 3 * 2 * 4),
    );
    const total = fromFixed(
      fromWords(energyWords[0] as number, (energyWords[1] as number) | 0),
    );
    const dispersion = force.useDispersionCorrection
      ? dispersionCorrection(
          force,
          (box[0] as number) * (box[1] as number) * (box[2] as number),
        )
      : 0;
    return {
      pairEnergy,
      exceptionEnergy: total - pairEnergy,
      dispersion,
      energy: total + dispersion,
      forces: unpackFixed(forceWords, n * 3),
    };
  } finally {
    for (const b of buffers) b.destroy();
  }
}
