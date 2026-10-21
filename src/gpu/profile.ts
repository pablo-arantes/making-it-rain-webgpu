/**
 * Per-kernel GPU timing, through `timestamp-query`.
 *
 * The engine runs at about 208 steps/s on 2373 atoms, which is roughly what
 * OpenMM's *CPU* platform manages on the same system — so there is headroom,
 * and nothing in the code says where it is. Two plausible culprits turned out
 * already to be fine: the readbacks live in the accessors rather than the
 * per-step path, and steps are batched into one submission. Guessing further
 * would be guessing, so this measures.
 *
 * 🔴 **GPU time, not wall-clock.** `performance.now()` around a submission
 * measures how long the *CPU* waited, which on a pipelined queue is neither
 * the kernel's duration nor a sum that adds up. Timestamps are written by the
 * GPU at pass boundaries and are the only thing that attributes cost to a
 * kernel.
 *
 * Optional by design. `timestamp-query` is in `OPTIONAL_FEATURES` because a
 * device may not offer it and the engine must not need it — so this returns
 * `undefined` rather than throwing, and a caller reports the absence instead
 * of failing. Nothing here may change a result: the profiler only adds
 * `timestampWrites` to passes that would run anyway.
 */

import type { Pass } from "./pipeline.js";

export interface PassTiming {
  readonly label: string;
  /** Nanoseconds of GPU time, summed over every occurrence of this label. */
  readonly nanoseconds: number;
  readonly calls: number;
}

export interface Profile {
  readonly passes: readonly PassTiming[];
  /** Summed GPU time across every pass, nanoseconds. */
  readonly totalNanoseconds: number;
  /** Wall-clock for the whole submission, for comparison with the sum. */
  readonly wallNanoseconds: number;
}

/** Whether a device can be profiled at all. */
export const canProfile = (device: GPUDevice): boolean =>
  device.features.has("timestamp-query");

/**
 * Run `passes` once with a timestamp around each, and report per-label GPU time.
 *
 * Returns `undefined` when the device has no `timestamp-query`, which is a
 * fact about the device rather than an error.
 *
 * `repeats` runs the whole set several times into one submission, because a
 * single pass over a small system is dominated by launch overhead — the thing
 * being measured is the steady-state cost, and the only honest way to see it is
 * to make the work large enough to dominate.
 */
export async function profilePasses(
  device: GPUDevice,
  label: string,
  passes: readonly Pass[],
  repeats = 1,
): Promise<Profile | undefined> {
  if (!canProfile(device)) return undefined;
  const count = passes.length * repeats;
  if (count === 0) {
    return { passes: [], totalNanoseconds: 0, wallNanoseconds: 0 };
  }

  // Two timestamps per pass, and the resolve buffer has to be a multiple of
  // 256 bytes for the copy that follows.
  const querySet = device.createQuerySet({ type: "timestamp", count: count * 2 });
  const resolved = device.createBuffer({
    size: Math.ceil((count * 2 * 8) / 256) * 256,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    label: `${label} timestamps`,
  });
  const readback = device.createBuffer({
    size: resolved.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: `${label} timestamps readback`,
  });

  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder({ label });
  const labels: string[] = [];
  for (let repeat = 0; repeat < repeats; repeat++) {
    for (const item of passes) {
      const slot = labels.length;
      labels.push(item.label);
      const pass = encoder.beginComputePass({
        label: item.label,
        timestampWrites: {
          querySet,
          beginningOfPassWriteIndex: slot * 2,
          endOfPassWriteIndex: slot * 2 + 1,
        },
      });
      pass.setPipeline(item.pipeline);
      if (item.dynamicOffsets === undefined) {
        pass.setBindGroup(0, item.bindGroup);
      } else {
        pass.setBindGroup(0, item.bindGroup, item.dynamicOffsets);
      }
      pass.dispatchWorkgroups(
        item.workgroups[0],
        item.workgroups[1] ?? 1,
        item.workgroups[2] ?? 1,
      );
      pass.end();
    }
  }
  encoder.resolveQuerySet(querySet, 0, count * 2, resolved, 0);
  encoder.copyBufferToBuffer(resolved, 0, readback, 0, resolved.size);

  const began = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const wallNanoseconds = (performance.now() - began) * 1e6;
  const error = await device.popErrorScope();
  if (error !== null) {
    querySet.destroy();
    resolved.destroy();
    readback.destroy();
    throw new Error(`${label}: ${error.message}`);
  }

  await readback.mapAsync(GPUMapMode.READ);
  // u64 nanoseconds. BigInt is the only lossless read, and the differences are
  // small enough to narrow to Number afterwards.
  const stamps = new BigInt64Array(readback.getMappedRange().slice(0));
  readback.unmap();

  const totals = new Map<string, { nanoseconds: number; calls: number }>();
  let totalNanoseconds = 0;
  for (const [slot, passLabel] of labels.entries()) {
    const start = stamps[slot * 2] as bigint;
    const end = stamps[slot * 2 + 1] as bigint;
    // 🔴 A zero or negative span is dropped rather than recorded. Timestamps
    // can come back as zero when a device quantises them coarsely or when the
    // queue reorders around an empty pass, and counting those as 0 ns would
    // silently flatter whichever kernel they landed on.
    if (end <= start) continue;
    const nanoseconds = Number(end - start);
    const existing = totals.get(passLabel);
    if (existing === undefined) totals.set(passLabel, { nanoseconds, calls: 1 });
    else {
      existing.nanoseconds += nanoseconds;
      existing.calls++;
    }
    totalNanoseconds += nanoseconds;
  }

  querySet.destroy();
  resolved.destroy();
  readback.destroy();

  return {
    passes: [...totals]
      .map(([passLabel, entry]) => ({ label: passLabel, ...entry }))
      .sort((a, b) => b.nanoseconds - a.nanoseconds),
    totalNanoseconds,
    wallNanoseconds,
  };
}

/** The breakdown as a share of the total, largest first. */
export function profileShares(
  profile: Profile,
): readonly { readonly label: string; readonly share: number; readonly nanoseconds: number }[] {
  if (profile.totalNanoseconds === 0) return [];
  return profile.passes.map((pass) => ({
    label: pass.label,
    share: pass.nanoseconds / profile.totalNanoseconds,
    nanoseconds: pass.nanoseconds,
  }));
}
