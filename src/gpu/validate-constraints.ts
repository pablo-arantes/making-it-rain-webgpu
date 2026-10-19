/**
 * The constraint kernels against the reference solver.
 *
 * Perturbs the fixture by a seeded displacement at the scale a 2 fs step
 * produces, projects it on both sides, and compares. Two things are reported
 * separately, because they fail differently: how far the device's answer is
 * from the reference's, and whether the device's answer actually satisfies the
 * constraints. The second is the one that matters physically, and a kernel can
 * fail it while sitting close to the reference.
 */

import { parseState } from "../engine/state.js";
import { parseSystem, type OpenMMSystem } from "../engine/system.js";
import { splitConstraints } from "../engine/constraint-clusters.js";
import { constrainPositions } from "../reference/constraints.js";
import { BufferPool } from "./allocator.js";
import { readableStorageUsage } from "./usage.js";
import { packConstraints, constrainOnDevice, uploadConstraintTables } from "./constrain.js";
import { packPositions, unpackFixed } from "./forces.js";
import { readBuffer } from "./compute.js";

export interface ConstraintComparison {
  readonly particleCount: number;
  readonly settleClusters: number;
  readonly iterativeClusters: number;
  readonly largestCluster: { readonly atoms: number; readonly constraints: number };
  /** nm, the worst per-component difference from the reference. */
  readonly worstDifference: number;
  /** The worst relative constraint error the device left behind. */
  readonly deviceWorstRelativeError: number;
  readonly referenceWorstRelativeError: number;
  readonly tolerance: number;
  readonly ok: boolean;
}

/** How far the worst constrained distance is from its target, relatively. */
function worstRelativeError(system: OpenMMSystem, positions: Float64Array): number {
  let worst = 0;
  for (let c = 0; c < system.constraintCount; c++) {
    const p1 = (system.constraintP1[c] as number) * 3;
    const p2 = (system.constraintP2[c] as number) * 3;
    let squared = 0;
    for (let axis = 0; axis < 3; axis++) {
      squared += ((positions[p1 + axis] as number) - (positions[p2 + axis] as number)) ** 2;
    }
    worst = Math.max(
      worst,
      Math.abs(Math.sqrt(squared) / (system.constraintDistance[c] as number) - 1),
    );
  }
  return worst;
}

/**
 * Thresholds for the constraint kernels.
 *
 * Looser than the forces' because two different things contribute: SETTLE's
 * geometry in f32 rather than f64, and a per-cluster iteration that stops when
 * its own cluster is converged where the reference stops when all of them are.
 * Both land inside the same relative tolerance band, which on a 0.1 nm bond is
 * 1e-7 nm wide.
 */
export const CONSTRAINT_THRESHOLDS = {
  /** nm, against the reference's answer. */
  difference: 2e-6,
  /** The device must satisfy the constraints as well as it was asked to. */
  relativeErrorFactor: 4,
} as const;

export async function validateConstraints(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
  tolerance = 1e-6,
): Promise<ConstraintComparison> {
  const system = parseSystem(texts.systemXml);
  const state = parseState(texts.stateXml);
  const split = splitConstraints(system);
  const tables = packConstraints(system, split);

  // A displacement at the scale a 2 fs step produces, which is the regime the
  // solver actually runs in.
  let seed = 0x5eed;
  const random = (): number => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  const trialPositions = Float64Array.from(
    state.positions,
    (value) => value + (random() * 2 - 1) * 2e-4,
  );

  // The reference, at the coordinates the device will see.
  const referenceWords = packPositions(state.positions);
  const reference = unpackFixed(referenceWords, state.positions.length);
  const trialWords = packPositions(trialPositions);
  const expected = unpackFixed(trialWords, trialPositions.length);
  constrainPositions(system, reference, expected, tolerance, split);

  const pool = new BufferPool(device);
  try {
    uploadConstraintTables(pool, tables);
    const referenceBuffer = pool.buffer({
      label: "reference",
      byteLength: referenceWords.byteLength,
      usage: readableStorageUsage(),
    });
    device.queue.writeBuffer(
      referenceBuffer,
      0,
      referenceWords as unknown as BufferSource,
    );
    const trialBuffer = pool.buffer({
      label: "trial",
      byteLength: trialWords.byteLength,
      usage: readableStorageUsage(),
    });
    device.queue.writeBuffer(trialBuffer, 0, trialWords as unknown as BufferSource);

    await constrainOnDevice(pool, tables, referenceBuffer, trialBuffer, { tolerance });

    const got = unpackFixed(
      new Uint32Array(await readBuffer(device, trialBuffer, trialWords.byteLength)),
      trialPositions.length,
    );
    let worstDifference = 0;
    for (let i = 0; i < got.length; i++) {
      worstDifference = Math.max(
        worstDifference,
        Math.abs((got[i] as number) - (expected[i] as number)),
      );
    }
    const deviceWorstRelativeError = worstRelativeError(system, got);
    const referenceWorstRelativeError = worstRelativeError(system, expected);
    return {
      particleCount: system.particleCount,
      settleClusters: split.settle.length,
      iterativeClusters: split.iterativeClusters.length,
      largestCluster: split.largestCluster,
      worstDifference,
      deviceWorstRelativeError,
      referenceWorstRelativeError,
      tolerance,
      ok:
        worstDifference < CONSTRAINT_THRESHOLDS.difference &&
        deviceWorstRelativeError <
          tolerance * CONSTRAINT_THRESHOLDS.relativeErrorFactor,
    };
  } finally {
    pool.destroy();
  }
}
