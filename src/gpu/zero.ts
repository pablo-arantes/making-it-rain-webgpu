/**
 * Clearing accumulators on the device.
 *
 * A step begins by zeroing the force accumulator, the energy and the PME charge
 * grid. `writeBuffer` would do it, but the host then allocates and sends a
 * zeroed array every step -- 642 KB for 3HTB's forces alone -- so this is a
 * pass instead, built once and encoded with the rest.
 */

import { pipelineCacheFor } from "./compute.js";
import { ZERO } from "./kernels.js";
import { BufferPool } from "./allocator.js";
import { uniformUsage } from "./usage.js";
import { submitPasses, type Pass, type PipelineCache } from "./pipeline.js";

/**
 * A pass that zeroes the first `words` words of `buffer`.
 *
 * `label` names the extent uniform in the pool, so two clears of different
 * sizes do not collide on it.
 */
export function planZero(
  pool: BufferPool,
  cache: PipelineCache,
  label: string,
  buffer: GPUBuffer,
  words: number,
): Promise<Pass> {
  const extent = pool.buffer({
    label: `zero/${label}/extent`,
    byteLength: 16,
    usage: uniformUsage(),
  });
  pool.device.queue.writeBuffer(extent, 0, new Uint32Array([words, 0, 0, 0]));
  return cache.pass(
    { label: `zero/${label}`, code: ZERO, entryPoint: "zero_words" },
    [
      { binding: 0, resource: { buffer } },
      { binding: 1, resource: { buffer: extent } },
    ],
    [Math.ceil(words / 64)],
  );
}

/** Zero a buffer now, for a single-configuration path. */
export async function zeroOnDevice(
  pool: BufferPool,
  label: string,
  buffer: GPUBuffer,
  words: number,
): Promise<void> {
  const pass = await planZero(pool, pipelineCacheFor(pool.device), label, buffer, words);
  await submitPasses(pool.device, `zero/${label}`, [pass]);
}
