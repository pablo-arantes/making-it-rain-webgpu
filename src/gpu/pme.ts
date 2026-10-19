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

import { dispatch, readBuffer, storageBuffer, uploadBuffer } from "./compute.js";
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
 * Run every stage of a 3D transform, ping-ponging between two buffers.
 *
 * Returns which of the two buffers holds the result. Exported so the FFT can
 * have its own differential test against `src/reference/fft.ts`, rather than
 * being covered only through PME end to end.
 */
export async function transform3d(
  device: GPUDevice,
  a: GPUBuffer,
  b: GPUBuffer,
  grid: Vec3,
  sign: number,
  plans: ReadonlyArray<FftPlan>,
): Promise<GPUBuffer> {
  let from = a;
  let to = b;
  for (const [axis, [length, stride, lineCount]] of axes(grid).entries()) {
    const plan = plans[axis] as FftPlan;
    for (const stage of plan.stages) {
      const uniform = new ArrayBuffer(24);
      const u32 = new Uint32Array(uniform);
      const f32 = new Float32Array(uniform);
      u32[0] = length;
      u32[1] = stride;
      u32[2] = lineCount;
      u32[3] = stage.radix;
      u32[4] = stage.ns;
      f32[5] = sign;
      const stageBuffer = uploadBuffer(device, {
        label: "fft stage",
        data: new Uint8Array(uniform),
        usage: GPUBufferUsage.UNIFORM,
      });
      try {
        await dispatch(device, {
          label: `fft radix ${stage.radix} ns ${stage.ns}`,
          code: FFT,
          entryPoint: "fft_stage",
          bindings: [
            { binding: 0, resource: { buffer: from } },
            { binding: 1, resource: { buffer: to } },
            { binding: 2, resource: { buffer: stageBuffer } },
          ],
          workgroups: [Math.ceil(((length / stage.radix) * lineCount) / 64)],
        });
      } finally {
        stageBuffer.destroy();
      }
      const swap = from;
      from = to;
      to = swap;
    }
  }
  return from;
}

/** Evaluate reciprocal-space PME on the device. */
export async function evaluateReciprocalSpace(
  device: GPUDevice,
  force: NonbondedForce,
  positionWords: Uint32Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  grid: Vec3,
): Promise<ReciprocalGpuResult> {
  const box = rectangularBox(boxVectors);
  const [nx, ny, nz] = grid as unknown as [number, number, number];
  const total = nx * ny * nz;
  const n = force.particleCount;
  const plans = [nz, ny, nx].map((length) => planFft(length));
  // `axes` returns z, y, x in that order, and so does `plans`.
  const stageCount = plans.reduce((sum, plan) => sum + plan.stages.length, 0);

  const volume = (box[0] as number) * (box[1] as number) * (box[2] as number);
  const prefactor = (ONE_4PI_EPS0 * 2 * Math.PI) / volume;

  const charges = Float32Array.from(force.charge);
  const moduli = Float32Array.from([
    ...bsplineModuli(nx, PME_ORDER),
    ...bsplineModuli(ny, PME_ORDER),
    ...bsplineModuli(nz, PME_ORDER),
  ]);

  const positionBuffer = uploadBuffer(device, {
    label: "pme positions",
    data: positionWords,
    usage: GPUBufferUsage.STORAGE,
  });
  const chargeBuffer = uploadBuffer(device, {
    label: "pme charges",
    data: charges,
    usage: GPUBufferUsage.STORAGE,
  });
  const fixedGrid = storageBuffer(device, "pme fixed grid", total * 2 * 4);
  device.queue.writeBuffer(fixedGrid, 0, new Uint32Array(total * 2));
  const gridA = storageBuffer(device, "pme grid a", total * 2 * 4);
  const gridB = storageBuffer(device, "pme grid b", total * 2 * 4);
  const modulusBuffer = uploadBuffer(device, {
    label: "pme moduli",
    data: moduli,
    usage: GPUBufferUsage.STORAGE,
  });
  const forceBuffer = storageBuffer(device, "pme forces", n * 3 * 2 * 4);
  device.queue.writeBuffer(forceBuffer, 0, new Uint32Array(n * 3 * 2));
  const energyBuffer = storageBuffer(device, "pme energy", 8);
  device.queue.writeBuffer(energyBuffer, 0, new Uint32Array(2));

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
  const pmeBuffer = uploadBuffer(device, {
    label: "pme params",
    data: new Uint8Array(pmeUniform),
    usage: GPUBufferUsage.UNIFORM,
  });
  const sizeBuffer = uploadBuffer(device, {
    label: "pme grid size",
    data: new Uint32Array([total]),
    usage: GPUBufferUsage.UNIFORM,
  });
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
  const convolveBuffer = uploadBuffer(device, {
    label: "pme convolution params",
    data: new Uint8Array(convolveUniform),
    usage: GPUBufferUsage.UNIFORM,
  });
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
  const interpolateBuffer = uploadBuffer(device, {
    label: "pme interpolation params",
    data: new Uint8Array(interpolateUniform),
    usage: GPUBufferUsage.UNIFORM,
  });

  const buffers = [
    positionBuffer, chargeBuffer, fixedGrid, gridA, gridB, modulusBuffer,
    forceBuffer, energyBuffer, pmeBuffer, sizeBuffer, convolveBuffer,
    interpolateBuffer,
  ];
  try {
    await dispatch(device, {
      label: "pme spread",
      code: PME_SPREAD,
      entryPoint: "spread_charge",
      bindings: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: chargeBuffer } },
        { binding: 2, resource: { buffer: fixedGrid } },
        { binding: 3, resource: { buffer: pmeBuffer } },
      ],
      workgroups: [Math.ceil(n / 64)],
    });
    await dispatch(device, {
      label: "pme load grid",
      code: PME_GRID,
      entryPoint: "load_grid",
      bindings: [
        { binding: 0, resource: { buffer: fixedGrid } },
        { binding: 1, resource: { buffer: gridA } },
        { binding: 2, resource: { buffer: sizeBuffer } },
      ],
      workgroups: [Math.ceil(total / 64)],
    });

    const transformed = await transform3d(device, gridA, gridB, grid, -1, plans);
    await dispatch(device, {
      label: "pme convolve",
      code: PME_CONVOLVE,
      entryPoint: "convolve",
      bindings: [
        { binding: 0, resource: { buffer: transformed } },
        { binding: 1, resource: { buffer: modulusBuffer } },
        { binding: 2, resource: { buffer: energyBuffer } },
        { binding: 3, resource: { buffer: convolveBuffer } },
      ],
      workgroups: [Math.ceil(total / 64)],
    });
    const other = transformed === gridA ? gridB : gridA;
    const potential = await transform3d(device, transformed, other, grid, 1, plans);

    await dispatch(device, {
      label: "pme interpolate",
      code: PME_INTERPOLATE,
      entryPoint: "interpolate_force",
      bindings: [
        { binding: 0, resource: { buffer: positionBuffer } },
        { binding: 1, resource: { buffer: forceBuffer } },
        { binding: 2, resource: { buffer: chargeBuffer } },
        { binding: 3, resource: { buffer: potential } },
        { binding: 4, resource: { buffer: interpolateBuffer } },
      ],
      workgroups: [Math.ceil(n / 64)],
    });

    const energyWords = new Uint32Array(await readBuffer(device, energyBuffer, 8));
    const forceWords = new Uint32Array(
      await readBuffer(device, forceBuffer, n * 3 * 2 * 4),
    );
    return {
      energy: fromFixed(
        fromWords(energyWords[0] as number, (energyWords[1] as number) | 0),
      ),
      forces: unpackFixed(forceWords, n * 3),
      grid,
      stageCount,
    };
  } finally {
    for (const b of buffers) b.destroy();
  }
}
