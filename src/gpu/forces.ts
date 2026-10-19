/**
 * Host side of the bonded force kernels.
 *
 * Two layers, and the second is written in terms of the first so there is only
 * ever one way these kernels get run.
 *
 * `prepareBonded` uploads a force's term tables into pool-owned buffers and
 * `planBonded` builds the pass that evaluates them. The tables depend on the
 * `System` and never on the coordinates, so a step loop does both once and then
 * encodes the pass every step.
 *
 * `evaluateBondedForce` is the single-configuration path the differential tests
 * use: make a pool, prepare, plan, submit, read back, drop the pool. It is six
 * lines of orchestration over the same two functions, which is the point --
 * a loop that evaluated forces differently from the tests would be a loop whose
 * forces are untested.
 */

import { readBuffer, pipelineCacheFor } from "./compute.js";
import { submitPasses, type Pass, type PipelineCache } from "./pipeline.js";
import { BufferPool } from "./allocator.js";
import { readableStorageUsage, storageUsage, uniformUsage } from "./usage.js";
import { BONDED } from "./kernels.js";
import { fromFixed, fromWords, toFixed } from "../reference/fixed-point.js";
import type {
  HarmonicAngleForce,
  HarmonicBondForce,
  PeriodicTorsionForce,
} from "../engine/system.js";

/** Positions as the kernels expect them: two fixed-point words per component. */
export function packPositions(positions: Float64Array): Uint32Array {
  const words = new Uint32Array(positions.length * 2);
  for (let i = 0; i < positions.length; i++) {
    const fixed = toFixed(positions[i] as number);
    // Low word unsigned, high word as its two's-complement bits.
    words[i * 2] = Number(fixed & 0xffffffffn);
    words[i * 2 + 1] = Number((fixed >> 32n) & 0xffffffffn);
  }
  return words;
}

/** Undo `packPositions`, exactly. */
export function unpackFixed(words: Uint32Array, count: number): Float64Array {
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = fromFixed(
      fromWords(words[i * 2] as number, (words[i * 2 + 1] as number) | 0),
    );
  }
  return out;
}

export interface BondedResult {
  /** kJ/mol */
  readonly energy: number;
  /** kJ/mol/nm, xyz interleaved */
  readonly forces: Float64Array;
}

export type BondedForce = HarmonicBondForce | HarmonicAngleForce | PeriodicTorsionForce;

const ENTRY_POINT = {
  HarmonicBondForce: "harmonic_bond",
  HarmonicAngleForce: "harmonic_angle",
  PeriodicTorsionForce: "periodic_torsion",
} as const;

/** The flat atom-index and parameter arrays each entry point reads. */
function packTerms(force: BondedForce): { atoms: Uint32Array; params: Float32Array } {
  switch (force.type) {
    case "HarmonicBondForce": {
      const atoms = new Uint32Array(force.count * 2);
      const params = new Float32Array(force.count * 2);
      for (let i = 0; i < force.count; i++) {
        atoms[i * 2] = force.p1[i] as number;
        atoms[i * 2 + 1] = force.p2[i] as number;
        params[i * 2] = force.length[i] as number;
        params[i * 2 + 1] = force.k[i] as number;
      }
      return { atoms, params };
    }
    case "HarmonicAngleForce": {
      const atoms = new Uint32Array(force.count * 3);
      const params = new Float32Array(force.count * 2);
      for (let i = 0; i < force.count; i++) {
        atoms[i * 3] = force.p1[i] as number;
        atoms[i * 3 + 1] = force.p2[i] as number;
        atoms[i * 3 + 2] = force.p3[i] as number;
        params[i * 2] = force.angle[i] as number;
        params[i * 2 + 1] = force.k[i] as number;
      }
      return { atoms, params };
    }
    case "PeriodicTorsionForce": {
      const atoms = new Uint32Array(force.count * 4);
      const params = new Float32Array(force.count * 3);
      for (let i = 0; i < force.count; i++) {
        atoms[i * 4] = force.p1[i] as number;
        atoms[i * 4 + 1] = force.p2[i] as number;
        atoms[i * 4 + 2] = force.p3[i] as number;
        atoms[i * 4 + 3] = force.p4[i] as number;
        params[i * 3] = force.periodicity[i] as number;
        params[i * 3 + 1] = force.phase[i] as number;
        params[i * 3 + 2] = force.k[i] as number;
      }
      return { atoms, params };
    }
  }
}

/** The buffers one bonded force needs, and how to dispatch it. */
export interface ResidentBonded {
  readonly type: BondedForce["type"];
  readonly termCount: number;
  readonly atoms: GPUBuffer;
  readonly params: GPUBuffer;
  readonly count: GPUBuffer;
}


/**
 * Upload one bonded force's term tables.
 *
 * `slot` distinguishes two forces of the same type, which an uploaded system
 * may well have -- the pool is idempotent by label, so without it the second
 * would silently reuse the first's buffers.
 */
export function prepareBonded(
  pool: BufferPool,
  force: BondedForce,
  slot = 0,
): ResidentBonded {
  const { atoms, params } = packTerms(force);
  const put = (name: string, data: ArrayBufferView, usage: number): GPUBuffer => {
    const buffer = pool.buffer({
      label: `bonded/${force.type}/${slot}/${name}`,
      byteLength: Math.max(4, data.byteLength),
      usage,
    });
    pool.device.queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
    return buffer;
  };
  return {
    type: force.type,
    termCount: force.count,
    atoms: put("atoms", atoms, storageUsage()),
    params: put("params", params, storageUsage()),
    count: put("count", new Uint32Array([force.count]), uniformUsage()),
  };
}

/** The pass that adds this force into the shared accumulators. */
export function planBonded(
  cache: PipelineCache,
  resident: ResidentBonded,
  accumulators: {
    readonly positions: GPUBuffer;
    readonly forces: GPUBuffer;
    readonly energy: GPUBuffer;
  },
): Promise<Pass> {
  return cache.pass(
    {
      label: `${resident.type} evaluate`,
      code: BONDED,
      entryPoint: ENTRY_POINT[resident.type],
    },
    // The indices accumulate.wgsl fixes: positions, forces, energy, then the
    // kernel's own data, then its uniform.
    [
      { binding: 0, resource: { buffer: accumulators.positions } },
      { binding: 1, resource: { buffer: accumulators.forces } },
      { binding: 2, resource: { buffer: accumulators.energy } },
      { binding: 3, resource: { buffer: resident.atoms } },
      { binding: 4, resource: { buffer: resident.params } },
      { binding: 5, resource: { buffer: resident.count } },
    ],
    [Math.max(1, Math.ceil(resident.termCount / 64))],
  );
}

/**
 * Evaluate one bonded force at one set of coordinates.
 *
 * `positionWords` comes from `packPositions`, so the caller can evaluate
 * several forces against one upload.
 */
export async function evaluateBondedForce(
  device: GPUDevice,
  force: BondedForce,
  positionWords: Uint32Array,
  particleCount: number,
): Promise<BondedResult> {
  const pool = new BufferPool(device);
  try {
    const components = particleCount * 3;
    const accumulators = {
      positions: pool.buffer({
        label: "positions",
        byteLength: positionWords.byteLength,
        usage: storageUsage(),
      }),
      // Two words per component, and both accumulators start at zero.
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
    const pass = await planBonded(cache, prepareBonded(pool, force), accumulators);
    await submitPasses(device, `${force.type} evaluate`, [pass]);

    const energyWords = new Uint32Array(
      await readBuffer(device, accumulators.energy, 8),
    );
    const forceWords = new Uint32Array(
      await readBuffer(device, accumulators.forces, components * 2 * 4),
    );
    return {
      energy: fromFixed(
        fromWords(energyWords[0] as number, (energyWords[1] as number) | 0),
      ),
      forces: unpackFixed(forceWords, components),
    };
  } finally {
    pool.destroy();
  }
}
