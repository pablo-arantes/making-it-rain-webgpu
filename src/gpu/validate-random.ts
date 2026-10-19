/**
 * The generator on the device against the generator on the host.
 *
 * Split in two deliberately. The **integer** stream must be bit-identical:
 * Threefry is add, xor and rotate, all of which wrap the same way everywhere,
 * and test/random.test.ts has already pinned the host's version to Random123's
 * published known answers. So a mismatch here localises to the WGSL
 * transcription and nothing else.
 *
 * The **deviates** cannot be bit-identical, because Box-Muller needs `log`,
 * `sqrt` and `cos`, which WGSL specifies to a few ulp and no further. What they
 * must be is close, and *how* close matters: the deviate is multiplied by
 * sqrt(kT/m) and lands in a velocity, so an outlier here shows up directly in a
 * thermostatted step.
 */

import {
  normalPair,
  thermostatNormal,
  threefry2x32,
  THERMOSTAT_DOMAIN,
} from "../reference/random.js";
import { dispatch, readBuffer } from "./compute.js";
import { RANDOM_PROBE } from "./kernels.js";
import { BufferPool } from "./allocator.js";


export interface RandomComparison {
  readonly particles: number;
  /** How many of the 64-bit raw outputs differ. Must be zero. */
  readonly bitMismatches: number;
  /** The worst absolute difference in a deviate. */
  readonly worstDeviate: number;
  /** Where that worst difference is, and what the two sides said. */
  readonly worstAt: { readonly particle: number; readonly axis: number };
  readonly worstHost: number;
  readonly worstDevice: number;
  /**
   * The radius Box-Muller used there.
   *
   * Reported because it is the quantity that explains an outlier: the radius is
   * `sqrt(-2 log u)`, and as u approaches 1 the logarithm cancels almost
   * completely, so a few ulp of error in `log` becomes a large relative error
   * in a small radius.
   */
  readonly worstRadius: number;
  readonly thresholds: { readonly deviate: number };
  readonly ok: boolean;
}

/** The tolerance a deviate has to meet. */
export const RANDOM_THRESHOLDS = { deviate: 1e-5 } as const;

export async function validateRandom(
  device: GPUDevice,
  particles = 8192,
  seed = 0x5eed,
  step = 0,
): Promise<RandomComparison> {
  const pool = new BufferPool(device);
  try {
    const bits = pool.buffer({
      label: "random/bits",
      byteLength: particles * 4 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const deviates = pool.buffer({
      label: "random/deviates",
      byteLength: particles * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const params = pool.buffer({
      label: "random/params",
      byteLength: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(params, 0, new Uint32Array([particles, seed, step, 0]));
    await dispatch(device, {
      label: "random/probe",
      code: RANDOM_PROBE,
      entryPoint: "probe",
      bindings: [
        { binding: 0, resource: { buffer: bits } },
        { binding: 1, resource: { buffer: deviates } },
        { binding: 2, resource: { buffer: params } },
      ],
      workgroups: [Math.ceil(particles / 64)],
    });

    const deviceBits = new Uint32Array(await readBuffer(device, bits, particles * 16));
    const deviceDeviates = new Float32Array(
      await readBuffer(device, deviates, particles * 12),
    );

    let bitMismatches = 0;
    for (let particle = 0; particle < particles; particle++) {
      for (let half = 0; half < 2; half++) {
        const expected = threefry2x32(
          [(particle * 2 + half) >>> 0, step >>> 0],
          [seed >>> 0, THERMOSTAT_DOMAIN],
        );
        const at = (particle * 2 + half) * 2;
        if (
          (deviceBits[at] as number) !== expected[0] ||
          (deviceBits[at + 1] as number) !== expected[1]
        ) {
          bitMismatches++;
        }
      }
    }

    let worstDeviate = 0;
    let worstAt = { particle: 0, axis: 0 };
    let worstHost = 0;
    let worstDevice = 0;
    for (let particle = 0; particle < particles; particle++) {
      for (let axis = 0; axis < 3; axis++) {
        const host = thermostatNormal(seed, step, particle, axis);
        const got = deviceDeviates[particle * 3 + axis] as number;
        const difference = Math.abs(got - host);
        if (difference > worstDeviate) {
          worstDeviate = difference;
          worstAt = { particle, axis };
          worstHost = host;
          worstDevice = got;
        }
      }
    }

    // The radius at the worst point, recomputed from the same counter.
    const half = worstAt.axis >> 1;
    const pair = normalPair(
      [(worstAt.particle * 2 + half) >>> 0, step >>> 0],
      [seed >>> 0, THERMOSTAT_DOMAIN],
    );
    const worstRadius = Math.hypot(pair[0], pair[1]);

    return {
      particles,
      bitMismatches,
      worstDeviate,
      worstAt,
      worstHost,
      worstDevice,
      worstRadius,
      thresholds: RANDOM_THRESHOLDS,
      ok: bitMismatches === 0 && worstDeviate < RANDOM_THRESHOLDS.deviate,
    };
  } finally {
    pool.destroy();
  }
}
