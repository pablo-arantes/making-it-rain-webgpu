/**
 * Throughput of the primitives the engine is built on.
 *
 * There is no engine to benchmark yet.  What there is are the two operations
 * every force kernel and every integrator step will be made of, and measuring
 * them now means a Phase 1 regression has something to be a regression from.
 *
 * Device-in, JSON-out, like `checks.ts`, so the same code serves
 * `npm run bench` and anything else that has a device.
 */

import { describeAdapter, type AdapterReport } from "./device.js";
import { accumulateFixedPoint, integratePosition } from "./precision.js";

export interface BenchRow {
  readonly name: string;
  readonly unit: string;
  readonly value: number;
  readonly detail: Readonly<Record<string, unknown>>;
}

export interface BenchReport {
  readonly adapter: AdapterReport;
  readonly rows: readonly BenchRow[];
}

/** Median of the middle of several timings, which is what to compare. */
async function timed(
  repeats: number,
  run: () => Promise<unknown>,
): Promise<{ medianMs: number; bestMs: number; timings: number[] }> {
  await run(); // warm up: the first call compiles the pipeline
  const timings: number[] = [];
  for (let i = 0; i < repeats; i++) {
    const t0 = performance.now();
    await run();
    timings.push(performance.now() - t0);
  }
  const sorted = [...timings].sort((a, b) => a - b);
  return {
    medianMs: sorted[Math.floor(sorted.length / 2)] as number,
    bestMs: sorted[0] as number,
    timings: timings.map((t) => Number(t.toFixed(3))),
  };
}

/**
 * Contended fixed-point accumulation, which is what a force kernel's scatter
 * path costs.
 *
 * Measured at three contention levels because that is the axis that decides
 * whether Phase 1 can afford atomics at all, or has to gather instead: one slot
 * is the worst case a kernel could ever hit, and 65536 slots is roughly the
 * per-atom case.
 */
export async function benchFixedPointAccumulate(
  device: GPUDevice,
): Promise<BenchRow[]> {
  const values = new Float32Array(1 << 20); // 1,048,576 contributions
  for (let i = 0; i < values.length; i++) {
    values[i] = Math.fround(Math.sin(i) * 1000);
  }
  const rows: BenchRow[] = [];
  for (const slots of [1, 4096, 65536]) {
    const { medianMs, bestMs, timings } = await timed(5, () =>
      accumulateFixedPoint(device, values, slots),
    );
    rows.push({
      name: `fixed-point accumulate, ${slots} slot${slots === 1 ? "" : "s"}`,
      unit: "million atomic adds per second",
      value: Number((values.length / 1e6 / (medianMs / 1000)).toFixed(1)),
      detail: {
        contributions: values.length,
        slots,
        contributionsPerSlot: values.length / slots,
        medianMs,
        bestMs,
        timings,
      },
    });
  }
  return rows;
}

/**
 * The compensated position integrator, one coordinate, serially.
 *
 * Serial on purpose: this measures the dependent-operation latency of the
 * integrator's inner loop, which is the thing a real step cannot parallelise
 * away.
 */
export async function benchPositionIntegrate(device: GPUDevice): Promise<BenchRow[]> {
  const steps = 1 << 20;
  const displacements = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    displacements[i] = Math.fround(Math.sin(i * 0.37) * 5e-4);
  }
  const { medianMs, bestMs, timings } = await timed(5, () =>
    integratePosition(device, 6.5, displacements),
  );
  return [
    {
      name: "position integrate, serial dependent chain",
      unit: "million steps per second",
      value: Number((steps / 1e6 / (medianMs / 1000)).toFixed(1)),
      detail: { steps, medianMs, bestMs, timings },
    },
  ];
}

export async function runBench(
  adapter: GPUAdapter,
  device: GPUDevice,
): Promise<BenchReport> {
  return {
    adapter: describeAdapter(adapter),
    rows: [
      ...(await benchFixedPointAccumulate(device)),
      ...(await benchPositionIntegrate(device)),
    ],
  };
}
