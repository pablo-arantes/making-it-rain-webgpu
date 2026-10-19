/**
 * Host side of the position restraints.
 *
 * Same shape as the bonded path: `prepareRestraint` uploads the terms, which
 * depend on the `System` and not on the coordinates, and `planRestraint` builds
 * the pass. A run's equilibration stage adds this force and its production
 * stage does not, so the loop takes the force list it is given rather than
 * assuming one.
 *
 * The reference positions go to the device in **fixed point**, like the
 * coordinates, so the kernel can difference and wrap them exactly. Converting
 * them to f32 first would throw away 4.8e-7 nm at the far edge of a 6.5 nm box
 * before the wrap could use it.
 */

import { pipelineCacheFor, readBuffer } from "./compute.js";
import { submitPasses } from "./pipeline.js";
import { packPositions, unpackFixed } from "./forces.js";
import { fromFixed, fromWords } from "../reference/fixed-point.js";
import { readableStorageUsage } from "./usage.js";
import { RESTRAINT } from "./kernels.js";
import { BufferPool } from "./allocator.js";
import { storageUsage, uniformUsage } from "./usage.js";
import { fixedWords } from "../reference/fixed-point.js";
import { writeFixedBox } from "./nonbonded.js";
import { rectangularBox } from "../reference/forces/nonbonded.js";
import type { Pass, PipelineCache } from "./pipeline.js";
import type { PositionRestraintForce, Vec3 } from "../engine/system.js";

export interface ResidentRestraint {
  readonly count: number;
  readonly terms: GPUBuffer;
  readonly reference: GPUBuffer;
  readonly params: GPUBuffer;
}

export function prepareRestraint(
  pool: BufferPool,
  force: PositionRestraintForce,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  slot = 0,
): ResidentRestraint {
  const box = rectangularBox(boxVectors);
  const count = Math.max(1, force.count);

  // Two words a term: the atom index, and `k` as raw bits.
  const terms = new Uint32Array(count * 2);
  const termFloats = new Float32Array(terms.buffer);
  for (let i = 0; i < force.count; i++) {
    terms[i * 2] = force.particle[i] as number;
    termFloats[i * 2 + 1] = force.k[i] as number;
  }

  // The reference positions, in the same fixed point the coordinates use.
  const reference = new Uint32Array(count * 3 * 2);
  for (let i = 0; i < force.count; i++) {
    const axes = [force.x0[i] as number, force.y0[i] as number, force.z0[i] as number];
    for (let axis = 0; axis < 3; axis++) {
      const words = fixedWords(axes[axis] as number);
      reference[(i * 3 + axis) * 2] = words.lo;
      reference[(i * 3 + axis) * 2 + 1] = words.hi;
    }
  }

  // 112 bytes, and the arithmetic is worth writing out because getting it
  // wrong cost a round trip: a `count` and three pads fill the first 16, then
  // `box` is three `vec4<u32>` at a 16-byte stride (16..63) and `half_box` is
  // three more (64..111). The first version allocated 64 bytes and wrote
  // `half_box` at offset 48, overlapping the box -- which WebGPU rejects as a
  // binding smaller than the struct, not as the overlap it actually was.
  const params = new Uint32Array(28);
  params[0] = force.count;
  writeFixedBox(params, 4, box);

  const put = (name: string, data: Uint32Array, usage: number): GPUBuffer => {
    const buffer = pool.buffer({
      label: `restraint/${slot}/${name}`,
      byteLength: data.byteLength,
      usage,
    });
    pool.device.queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
    return buffer;
  };
  return {
    count: force.count,
    terms: put("terms", terms, storageUsage()),
    reference: put("reference", reference, storageUsage()),
    params: put("params", params, uniformUsage()),
  };
}

export function planRestraint(
  cache: PipelineCache,
  resident: ResidentRestraint,
  accumulators: {
    readonly positions: GPUBuffer;
    readonly forces: GPUBuffer;
    readonly energy: GPUBuffer;
  },
): Promise<Pass> {
  return cache.pass(
    {
      label: "position restraint",
      code: RESTRAINT,
      entryPoint: "position_restraint",
    },
    [
      { binding: 0, resource: { buffer: accumulators.positions } },
      { binding: 1, resource: { buffer: accumulators.forces } },
      { binding: 2, resource: { buffer: accumulators.energy } },
      { binding: 3, resource: { buffer: resident.terms } },
      { binding: 4, resource: { buffer: resident.reference } },
      { binding: 5, resource: { buffer: resident.params } },
    ],
    [Math.max(1, Math.ceil(resident.count / 64))],
  );
}

/**
 * Evaluate the restraint alone at one configuration.
 *
 * Isolated the way `evaluateBondedForce` isolates one bonded term: a shared
 * accumulator holding every force would have to be compared against a
 * difference of two large nearly-cancelling totals, where a restraint of a few
 * kJ/mol would be invisible. OpenMM's side is isolated too, by giving the
 * restraint its own force group.
 */
export async function evaluateRestraint(
  device: GPUDevice,
  force: PositionRestraintForce,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  particleCount: number,
): Promise<{ energy: number; forces: Float64Array }> {
  const pool = new BufferPool(device);
  try {
    const components = particleCount * 3;
    const accumulators = {
      positions: pool.buffer({
        label: "restraint/positions",
        byteLength: components * 8,
        usage: storageUsage(),
      }),
      forces: pool.buffer({
        label: "restraint/forces",
        byteLength: components * 2 * 4,
        usage: readableStorageUsage(),
      }),
      energy: pool.buffer({
        label: "restraint/energy",
        byteLength: 8,
        usage: readableStorageUsage(),
      }),
    };
    device.queue.writeBuffer(
      accumulators.positions,
      0,
      packPositions(positions) as unknown as BufferSource,
    );
    device.queue.writeBuffer(accumulators.forces, 0, new Uint32Array(components * 2));
    device.queue.writeBuffer(accumulators.energy, 0, new Uint32Array(2));

    const resident = prepareRestraint(pool, force, boxVectors);
    const pass = await planRestraint(pipelineCacheFor(device), resident, accumulators);
    await submitPasses(device, "position restraint", [pass]);

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
