/**
 * Host orchestration of reciprocal-space PME on the device.
 *
 * Five stages: spread the charges onto a fixed-point mesh, convert it to a
 * complex grid, transform it, convolve, transform back, and interpolate the
 * result onto the atoms as forces.  The transform is a sequence of Stockham
 * stages, one dispatch each, ping-ponging between two grids -- the plan comes
 * from `src/reference/fft.ts`, which is the same algorithm in f64 and is what
 * the kernel is tested against.
 */

import { pipelineCacheFor, readBuffer } from "./compute.js";
import { submitPasses, type Pass, type PipelineCache } from "./pipeline.js";
import { BufferPool } from "./allocator.js";
import { readableStorageUsage, storageUsage, uniformUsage } from "./usage.js";
import { planZero } from "./zero.js";
import {
  FFT,
  PME_CONVOLVE,
  PME_GRID,
  PME_INTERPOLATE,
  PME_SPREAD,
} from "./kernels.js";
import { unpackFixed } from "./forces.js";
import { fromFixed, fromWords } from "../reference/fixed-point.js";
import { planFft, type FftPlan } from "../reference/fft.js";
import { bsplineModuli } from "../reference/forces/bspline.js";
import { PME_ORDER } from "../reference/forces/pme.js";
import { rectangularBox } from "../reference/forces/nonbonded.js";
import { ONE_4PI_EPS0 } from "../engine/units.js";
import type { NonbondedForce, Vec3 } from "../engine/system.js";

export interface ReciprocalGpuResult {
  /** The mesh sum alone, kJ/mol: no self energy, no net-charge correction. */
  readonly energy: number;
  /** kJ/mol/nm, xyz interleaved. */
  readonly forces: Float64Array;
  readonly grid: Vec3;
  /** How many FFT dispatches the transform took, each way. */
  readonly stageCount: number;
}

/** The three axes of a 3D transform, as (length, stride, lineCount). */
export function axes(grid: Vec3): ReadonlyArray<readonly [number, number, number]> {
  const [nx, ny, nz] = grid as unknown as [number, number, number];
  const total = nx * ny * nz;
  return [
    [nz, 1, total / nz],
    [ny, nz, total / ny],
    [nx, ny * nz, total / nx],
  ];
}

/**
 * Plan every stage of a 3D transform, ping-ponging between two buffers.
 *
 * The schedule is a pure function of the grid: which axis, how long, what
 * stride, which radix, and therefore how many swaps -- so which of the two
 * buffers ends up holding the result is known before anything runs. That is
 * what lets a transform be a list of passes rather than a sequence of
 * dispatches, and `resultIsA` is how the caller knows where to look.
 *
 * One uniform buffer per stage, from the pool, because a bind group holds a
 * buffer rather than its contents: thirty-odd 24-byte uniforms is the price of
 * never rewriting one mid-step.
 */
export interface TransformPlan {
  readonly passes: readonly Pass[];
  readonly resultIsA: boolean;
}

export async function planTransform3d(
  pool: BufferPool,
  cache: PipelineCache,
  a: GPUBuffer,
  b: GPUBuffer,
  grid: Vec3,
  sign: number,
  plans: ReadonlyArray<FftPlan>,
  label: string,
): Promise<TransformPlan> {
  const passes: Pass[] = [];
  let fromIsA = true;
  for (const [axis, [length, stride, lineCount]] of axes(grid).entries()) {
    const plan = plans[axis] as FftPlan;
    for (const [index, stage] of plan.stages.entries()) {
      const uniform = new ArrayBuffer(24);
      const u32 = new Uint32Array(uniform);
      const f32 = new Float32Array(uniform);
      u32[0] = length;
      u32[1] = stride;
      u32[2] = lineCount;
      u32[3] = stage.radix;
      u32[4] = stage.ns;
      f32[5] = sign;
      const stageBuffer = pool.buffer({
        label: `fft/${label}/${axis}/${index}`,
        byteLength: 24,
        usage: uniformUsage(),
      });
      pool.device.queue.writeBuffer(stageBuffer, 0, new Uint8Array(uniform));
      passes.push(
        await cache.pass(
          {
            label: `fft radix ${stage.radix} ns ${stage.ns}`,
            code: FFT,
            entryPoint: "fft_stage",
          },
          [
            { binding: 0, resource: { buffer: fromIsA ? a : b } },
            { binding: 1, resource: { buffer: fromIsA ? b : a } },
            { binding: 2, resource: { buffer: stageBuffer } },
          ],
          [Math.ceil(((length / stage.radix) * lineCount) / 64)],
        ),
      );
      fromIsA = !fromIsA;
    }
  }
  return { passes, resultIsA: fromIsA };
}

/**
 * Run every stage of a 3D transform and return the buffer holding the result.
 *
 * Exported so the FFT can have its own differential test against
 * src/reference/fft.ts rather than being covered only through PME end to end.
 */
export async function transform3d(
  device: GPUDevice,
  a: GPUBuffer,
  b: GPUBuffer,
  grid: Vec3,
  sign: number,
  plans: ReadonlyArray<FftPlan>,
): Promise<GPUBuffer> {
  const pool = new BufferPool(device);
  try {
    const planned = await planTransform3d(
      pool,
      pipelineCacheFor(device),
      a,
      b,
      grid,
      sign,
      plans,
      `standalone${sign < 0 ? "Forward" : "Inverse"}`,
    );
    await submitPasses(device, "fft", planned.passes);
    return planned.resultIsA ? a : b;
  } finally {
    pool.destroy();
  }
}

/** Everything reciprocal space needs that the coordinates do not change. */
export interface ResidentPme {
  readonly grid: Vec3;
  readonly total: number;
  readonly particleCount: number;
  readonly stageCount: number;
  readonly charges: GPUBuffer;
  readonly fixedGrid: GPUBuffer;
  readonly gridA: GPUBuffer;
  readonly gridB: GPUBuffer;
  readonly moduli: GPUBuffer;
  readonly pmeParams: GPUBuffer;
  readonly sizeParams: GPUBuffer;
  readonly convolveParams: GPUBuffer;
  readonly interpolateParams: GPUBuffer;
  readonly plans: ReadonlyArray<FftPlan>;
}

export function preparePme(
  pool: BufferPool,
  force: NonbondedForce,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  grid: Vec3,
): ResidentPme {
  const device = pool.device;
  const box = rectangularBox(boxVectors);
  const [nx, ny, nz] = grid as unknown as [number, number, number];
  const total = nx * ny * nz;
  const n = force.particleCount;
  // `axes` returns z, y, x in that order, and so does this.
  const plans = [nz, ny, nx].map((length) => planFft(length));
  const stageCount = plans.reduce((sum, plan) => sum + plan.stages.length, 0);

  const volume = (box[0] as number) * (box[1] as number) * (box[2] as number);
  const prefactor = (ONE_4PI_EPS0 * 2 * Math.PI) / volume;

  const put = (name: string, data: ArrayBufferView, usage: number): GPUBuffer => {
    const buffer = pool.buffer({
      label: `pme/${name}`,
      byteLength: Math.max(4, data.byteLength),
      usage,
    });
    device.queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
    return buffer;
  };
  const blank = (name: string, byteLength: number): GPUBuffer =>
    pool.buffer({ label: `pme/${name}`, byteLength, usage: readableStorageUsage() });

  // struct Pme { u32 count; vec3<u32> grid; vec3<f32> box; f32 alpha; f32 prefactor }
  // vec3 aligns to 16, so grid is at 16, box at 32, alpha at 44, prefactor at 48.
  const pmeUniform = new ArrayBuffer(64);
  const pmeU32 = new Uint32Array(pmeUniform);
  const pmeF32 = new Float32Array(pmeUniform);
  pmeU32[0] = n;
  pmeU32[4] = nx;
  pmeU32[5] = ny;
  pmeU32[6] = nz;
  pmeF32[8] = box[0] as number;
  pmeF32[9] = box[1] as number;
  pmeF32[10] = box[2] as number;
  pmeF32[11] = alpha;
  pmeF32[12] = prefactor;

  // struct Convolution { vec3<u32> grid; vec2<u32> offsets; vec3<f32> box;
  //                      f32 alpha; f32 prefactor } -> grid 0, offsets 16,
  // box 32, alpha 44, prefactor 48.
  const convolveUniform = new ArrayBuffer(64);
  const convolveU32 = new Uint32Array(convolveUniform);
  const convolveF32 = new Float32Array(convolveUniform);
  convolveU32[0] = nx;
  convolveU32[1] = ny;
  convolveU32[2] = nz;
  convolveU32[4] = nx; // where the y moduli begin
  convolveU32[5] = nx + ny; // where the z moduli begin
  convolveF32[8] = box[0] as number;
  convolveF32[9] = box[1] as number;
  convolveF32[10] = box[2] as number;
  convolveF32[11] = alpha;
  convolveF32[12] = prefactor;

  // struct Pme { u32 count; vec3<u32> grid; vec3<f32> box } -> 48 bytes.
  const interpolateUniform = new ArrayBuffer(48);
  const interpolateU32 = new Uint32Array(interpolateUniform);
  const interpolateF32 = new Float32Array(interpolateUniform);
  interpolateU32[0] = n;
  interpolateU32[4] = nx;
  interpolateU32[5] = ny;
  interpolateU32[6] = nz;
  interpolateF32[8] = box[0] as number;
  interpolateF32[9] = box[1] as number;
  interpolateF32[10] = box[2] as number;

  return {
    grid,
    total,
    particleCount: n,
    stageCount,
    plans,
    charges: put("charges", Float32Array.from(force.charge), storageUsage()),
    fixedGrid: blank("fixedGrid", total * 2 * 4),
    gridA: blank("gridA", total * 2 * 4),
    gridB: blank("gridB", total * 2 * 4),
    moduli: put(
      "moduli",
      Float32Array.from([
        ...bsplineModuli(nx, PME_ORDER),
        ...bsplineModuli(ny, PME_ORDER),
        ...bsplineModuli(nz, PME_ORDER),
      ]),
      storageUsage(),
    ),
    pmeParams: put("params", new Uint8Array(pmeUniform), uniformUsage()),
    sizeParams: put("size", new Uint32Array([total]), uniformUsage()),
    convolveParams: put("convolveParams", new Uint8Array(convolveUniform), uniformUsage()),
    interpolateParams: put(
      "interpolateParams",
      new Uint8Array(interpolateUniform),
      uniformUsage(),
    ),
  };
}

/**
 * Every reciprocal-space pass, in order.
 *
 * Spread the charges, copy the fixed-point grid into a complex one, transform,
 * convolve, transform back, interpolate the forces. The charge grid is cleared
 * first because `spread_charge` accumulates into it, and the clear is a pass
 * rather than a host copy for the same reason everything else here is.
 */
export async function planPme(
  pool: BufferPool,
  cache: PipelineCache,
  resident: ResidentPme,
  accumulators: {
    readonly positions: GPUBuffer;
    readonly forces: GPUBuffer;
    readonly energy: GPUBuffer;
  },
): Promise<Pass[]> {
  const { grid, total, particleCount, plans } = resident;
  const passes: Pass[] = [
    await planZero(pool, cache, "pmeGrid", resident.fixedGrid, total * 2),
    await cache.pass(
      { label: "pme spread", code: PME_SPREAD, entryPoint: "spread_charge" },
      [
        { binding: 0, resource: { buffer: accumulators.positions } },
        { binding: 1, resource: { buffer: resident.charges } },
        { binding: 2, resource: { buffer: resident.fixedGrid } },
        { binding: 3, resource: { buffer: resident.pmeParams } },
      ],
      [Math.ceil(particleCount / 64)],
    ),
    await cache.pass(
      { label: "pme load grid", code: PME_GRID, entryPoint: "load_grid" },
      [
        { binding: 0, resource: { buffer: resident.fixedGrid } },
        { binding: 1, resource: { buffer: resident.gridA } },
        { binding: 2, resource: { buffer: resident.sizeParams } },
      ],
      [Math.ceil(total / 64)],
    ),
  ];

  const forward = await planTransform3d(
    pool,
    cache,
    resident.gridA,
    resident.gridB,
    grid,
    -1,
    plans,
    "forward",
  );
  passes.push(...forward.passes);
  const transformed = forward.resultIsA ? resident.gridA : resident.gridB;
  const other = forward.resultIsA ? resident.gridB : resident.gridA;

  passes.push(
    await cache.pass(
      { label: "pme convolve", code: PME_CONVOLVE, entryPoint: "convolve" },
      [
        { binding: 0, resource: { buffer: transformed } },
        { binding: 1, resource: { buffer: resident.moduli } },
        { binding: 2, resource: { buffer: accumulators.energy } },
        { binding: 3, resource: { buffer: resident.convolveParams } },
      ],
      [Math.ceil(total / 64)],
    ),
  );

  const inverse = await planTransform3d(
    pool,
    cache,
    transformed,
    other,
    grid,
    1,
    plans,
    "inverse",
  );
  passes.push(...inverse.passes);
  const potential = inverse.resultIsA ? transformed : other;

  passes.push(
    await cache.pass(
      {
        label: "pme interpolate",
        code: PME_INTERPOLATE,
        entryPoint: "interpolate_force",
      },
      [
        { binding: 0, resource: { buffer: accumulators.positions } },
        { binding: 1, resource: { buffer: accumulators.forces } },
        { binding: 2, resource: { buffer: resident.charges } },
        { binding: 3, resource: { buffer: potential } },
        { binding: 4, resource: { buffer: resident.interpolateParams } },
      ],
      [Math.ceil(particleCount / 64)],
    ),
  );
  return passes;
}

/** Evaluate reciprocal-space PME at one configuration. */
export async function evaluateReciprocalSpace(
  device: GPUDevice,
  force: NonbondedForce,
  positionWords: Uint32Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  grid: Vec3,
): Promise<ReciprocalGpuResult> {
  const pool = new BufferPool(device);
  try {
    const n = force.particleCount;
    const components = n * 3;
    const accumulators = {
      positions: pool.buffer({
        label: "pme/positions",
        byteLength: positionWords.byteLength,
        usage: storageUsage(),
      }),
      forces: pool.buffer({
        label: "pme/forces",
        byteLength: components * 2 * 4,
        usage: readableStorageUsage(),
      }),
      energy: pool.buffer({
        label: "pme/energy",
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
    const resident = preparePme(pool, force, boxVectors, alpha, grid);
    const passes = await planPme(pool, cache, resident, accumulators);
    await submitPasses(device, "pme", passes);

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
      grid,
      stageCount: resident.stageCount,
    };
  } finally {
    pool.destroy();
  }
}
