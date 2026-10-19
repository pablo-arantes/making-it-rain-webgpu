/**
 * Host side of the bonded force kernels.
 *
 * Uploads a parsed `System`'s bonded terms, runs one dispatch per term type,
 * and reads back the fixed-point forces and energy.  Phase 1 will keep the
 * buffers resident across steps; this evaluates one configuration, which is
 * what the differential tests need and what a single-point energy check is.
 */

import { dispatch, readBuffer, storageBuffer, uploadBuffer } from "./compute.js";
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

type BondedForce = HarmonicBondForce | HarmonicAngleForce | PeriodicTorsionForce;

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

/**
 * Evaluate one bonded force on the device.
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
  const { atoms, params } = packTerms(force);
  const positionBuffer = uploadBuffer(device, {
    label: "positions",
    data: positionWords,
    usage: GPUBufferUsage.STORAGE,
  });
  const atomBuffer = uploadBuffer(device, {
    label: `${force.type} atoms`,
    data: atoms,
    usage: GPUBufferUsage.STORAGE,
  });
  const paramBuffer = uploadBuffer(device, {
    label: `${force.type} params`,
    data: params,
    usage: GPUBufferUsage.STORAGE,
  });
  // Two words per component, and the buffer starts at zero.
  const forceBytes = particleCount * 3 * 2 * 4;
  const forceBuffer = storageBuffer(device, "forces", forceBytes);
  device.queue.writeBuffer(forceBuffer, 0, new Uint32Array(particleCount * 3 * 2));
  const energyBuffer = storageBuffer(device, "energy", 8);
  device.queue.writeBuffer(energyBuffer, 0, new Uint32Array(2));
  const countBuffer = uploadBuffer(device, {
    label: `${force.type} count`,
    data: new Uint32Array([force.count]),
    usage: GPUBufferUsage.UNIFORM,
  });

  try {
    await dispatch(device, {
      label: `${force.type} evaluate`,
      code: BONDED,
      entryPoint: ENTRY_POINT[force.type],
      // The order accumulate.wgsl fixes: positions, forces, energy, then the
      // kernel's own data, then its uniform.
      bindings: [
        { buffer: positionBuffer },
        { buffer: forceBuffer },
        { buffer: energyBuffer },
        { buffer: atomBuffer },
        { buffer: paramBuffer },
        { buffer: countBuffer },
      ],
      workgroups: [Math.max(1, Math.ceil(force.count / 64))],
    });
    const energyWords = new Uint32Array(await readBuffer(device, energyBuffer, 8));
    const forceWords = new Uint32Array(await readBuffer(device, forceBuffer, forceBytes));
    return {
      energy: fromFixed(
        fromWords(energyWords[0] as number, (energyWords[1] as number) | 0),
      ),
      forces: unpackFixed(forceWords, particleCount * 3),
    };
  } finally {
    for (const b of [
      positionBuffer,
      atomBuffer,
      paramBuffer,
      forceBuffer,
      energyBuffer,
      countBuffer,
    ]) {
      b.destroy();
    }
  }
}
