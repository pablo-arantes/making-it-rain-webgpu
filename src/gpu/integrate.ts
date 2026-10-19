/**
 * Host side of the integrators and the whole-system reductions.
 *
 * Both integrators need the constraint solver to run *inside* the step, so a
 * step is several dispatches with constraint dispatches between them. That
 * interleaving is what this file owns; the order is the contract, and getting
 * it wrong gives a result that is wrong by far less than it looks -- which is
 * why `src/gpu/validate-integrator.ts` compares against the reference step
 * rather than against a tolerance someone chose.
 *
 * Every buffer comes from the pool and is named, so a step allocates nothing.
 * What this does still do per dispatch is build a shader module, a pipeline and
 * a bind group, and then wait for the queue to drain -- six sync points a step.
 * That is deliberate for now: it keeps a validation run easy to follow, and the
 * step loop replaces it with cached pipelines and one submission.
 */

import { dispatch, readBuffer } from "./compute.js";
import { INTEGRATE, REDUCE } from "./kernels.js";
import { BufferPool } from "./allocator.js";
import { constrainOnDevice, type ConstraintTables } from "./constrain.js";
import { fromFixed, fromWords } from "../reference/fixed-point.js";
import { BOLTZMANN_KJ_PER_MOL_K } from "../engine/units.js";
import type { OpenMMSystem } from "../engine/system.js";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

/** Bit 0 of `StepParams.flags`. */
const DRAW_NOISE = 1;

/**
 * The bind group layouts, declared rather than derived.
 *
 * Both modules hold several entry points over one set of buffers -- the
 * integrator's five phases read overlapping subsets of the same seven -- and
 * `layout: "auto"` derives a *separate* layout for each from what it reaches,
 * so a bind group built for the whole set is rejected by all of them. Declaring
 * the layout once makes every phase share it, which is also what lets the step
 * loop build one bind group and reuse it.
 *
 * Cached per device: a layout is immutable and creating one per dispatch would
 * be pure waste.
 */
const layouts = new WeakMap<GPUDevice, { integrate: GPUBindGroupLayout; reduce: GPUBindGroupLayout }>();

type Kind = "storage" | "read-only-storage" | "uniform";

function makeLayout(
  device: GPUDevice,
  label: string,
  kinds: readonly Kind[],
): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    label,
    entries: kinds.map((kind, binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: kind },
    })),
  });
}

function layoutsFor(device: GPUDevice): {
  integrate: GPUBindGroupLayout;
  reduce: GPUBindGroupLayout;
} {
  const existing = layouts.get(device);
  if (existing !== undefined) return existing;
  const made = {
    // positions, velocities, forces, trial, unconstrained, residual,
    // inverseMass, params -- in the order integrate.wgsl declares them.
    integrate: makeLayout(device, "integrate", [
      "storage",
      "storage",
      "read-only-storage",
      "storage",
      "storage",
      "storage",
      "read-only-storage",
      "uniform",
    ]),
    // velocities, mass, totals, params.
    reduce: makeLayout(device, "reduce", [
      "storage",
      "read-only-storage",
      "storage",
      "uniform",
    ]),
  };
  layouts.set(device, made);
  return made;
}

/** The buffers a step reads and writes, all owned by the pool. */
export interface IntegratorBuffers {
  readonly positions: GPUBuffer;
  readonly velocities: GPUBuffer;
  readonly forces: GPUBuffer;
  readonly trial: GPUBuffer;
  readonly unconstrained: GPUBuffer;
  readonly residual: GPUBuffer;
  readonly totals: GPUBuffer;
}

/**
 * Create the step's buffers.
 *
 * `uploadConstraintTables` must have run first: the masses live under
 * `system/mass` and `system/inverseMass`, which both the constraint kernels and
 * these share.
 */
export function prepareIntegrator(
  pool: BufferPool,
  system: OpenMMSystem,
): IntegratorBuffers {
  const components = system.particleCount * 3;
  const named = (label: string, byteLength: number): GPUBuffer =>
    pool.buffer({ label, byteLength, usage: STORAGE });
  return {
    positions: named("integrate/positions", components * 8),
    velocities: named("integrate/velocities", components * 4),
    forces: named("integrate/forces", components * 8),
    trial: named("integrate/trial", components * 8),
    unconstrained: named("integrate/unconstrained", components * 8),
    residual: named("integrate/residual", components * 4),
    totals: named("integrate/totals", 24),
  };
}

export interface IntegratorOptions {
  /** ps. */
  readonly timeStep: number;
  readonly constraintTolerance: number;
  /** K. Only read when there is friction. */
  readonly temperature?: number;
  /** ps^-1. Zero makes the thermostat a no-op and the step deterministic. */
  readonly friction?: number;
  /** Required once friction is non-zero: a run must record its seed. */
  readonly seed?: number;
  /** Which step this is, the generator's other counter word. */
  readonly step?: number;
}

/**
 * Pack `StepParams`.
 *
 * Eight scalars, so no vec3 padding to get wrong -- the one alignment trap
 * WGSL uniforms set, and the reason this struct has none.
 */
function stepParams(
  pool: BufferPool,
  system: OpenMMSystem,
  options: IntegratorOptions,
): GPUBuffer {
  const friction = options.friction ?? 0;
  const temperature = options.temperature ?? 0;
  const velocityScale = Math.exp(-options.timeStep * friction);
  const noiseScale = Math.sqrt(1 - velocityScale * velocityScale);
  if (noiseScale > 0 && options.seed === undefined) {
    throw new Error(
      "an integrator step with friction needs a seed; a run must record one " +
        "so that it can be repeated. Pass `seed`.",
    );
  }
  const bytes = new ArrayBuffer(32);
  new Uint32Array(bytes, 0, 4).set([
    system.particleCount,
    options.seed ?? 0,
    options.step ?? 0,
    noiseScale > 0 ? DRAW_NOISE : 0,
  ]);
  new Float32Array(bytes, 16, 4).set([
    options.timeStep,
    velocityScale,
    noiseScale,
    BOLTZMANN_KJ_PER_MOL_K * temperature,
  ]);
  const buffer = pool.buffer({
    label: "integrate/params",
    byteLength: 32,
    usage: UNIFORM,
  });
  pool.device.queue.writeBuffer(buffer, 0, new Uint8Array(bytes));
  return buffer;
}

function integratorBindings(
  buffers: IntegratorBuffers,
  pool: BufferPool,
  params: GPUBuffer,
): readonly { binding: number; resource: GPUBindingResource }[] {
  const inverseMass = pool.existing("system/inverseMass");
  if (inverseMass === undefined) {
    throw new Error(
      "system/inverseMass is not in the pool; call uploadConstraintTables first",
    );
  }
  return [
    { binding: 0, resource: { buffer: buffers.positions } },
    { binding: 1, resource: { buffer: buffers.velocities } },
    { binding: 2, resource: { buffer: buffers.forces } },
    { binding: 3, resource: { buffer: buffers.trial } },
    { binding: 4, resource: { buffer: buffers.unconstrained } },
    { binding: 5, resource: { buffer: buffers.residual } },
    { binding: 6, resource: { buffer: inverseMass } },
    { binding: 7, resource: { buffer: params } },
  ];
}

/**
 * Run one phase of a step.
 *
 * Exported so a diagnostic can drive the phases one at a time and read the
 * state between them; a step itself goes through `verletStepOnDevice` or
 * `langevinStepOnDevice`, which own the order.
 */
export const dispatchPhase = async (
  pool: BufferPool,
  entryPoint: string,
  buffers: IntegratorBuffers,
  system: OpenMMSystem,
  options: IntegratorOptions,
): Promise<void> =>
  phase(
    pool,
    entryPoint,
    integratorBindings(buffers, pool, stepParams(pool, system, options)),
    system.particleCount,
  );

const phase = async (
  pool: BufferPool,
  entryPoint: string,
  bindings: readonly { binding: number; resource: GPUBindingResource }[],
  particleCount: number,
): Promise<void> =>
  dispatch(pool.device, {
    label: `integrate/${entryPoint}`,
    code: INTEGRATE,
    entryPoint,
    bindings,
    bindGroupLayout: layoutsFor(pool.device).integrate,
    workgroups: [Math.ceil(particleCount / 64)],
  });

/**
 * One velocity-Verlet step: kick and drift, SHAKE, then take the velocity from
 * the constrained displacement.
 */
export async function verletStepOnDevice(
  pool: BufferPool,
  system: OpenMMSystem,
  tables: ConstraintTables,
  buffers: IntegratorBuffers,
  options: IntegratorOptions,
): Promise<void> {
  const bindings = integratorBindings(buffers, pool, stepParams(pool, system, options));
  await phase(pool, "verlet_kick_drift", bindings, system.particleCount);
  await constrainOnDevice(pool, tables, buffers.positions, buffers.trial, {
    tolerance: options.constraintTolerance,
  });
  await phase(pool, "verlet_finish", bindings, system.particleCount);
}

/**
 * One LFMiddle step, which is what `LangevinIntegrator` has been since OpenMM
 * 8.2 and so what the notebook runs.
 *
 * Note the velocity projection after the kick. Verlet does not do it, and it is
 * one of only two things that distinguish the two integrators at zero friction
 * -- the other being which kinetic energy they report.
 */
export async function langevinStepOnDevice(
  pool: BufferPool,
  system: OpenMMSystem,
  tables: ConstraintTables,
  buffers: IntegratorBuffers,
  options: IntegratorOptions,
): Promise<void> {
  const bindings = integratorBindings(buffers, pool, stepParams(pool, system, options));
  await phase(pool, "lfmiddle_kick", bindings, system.particleCount);
  await constrainOnDevice(pool, tables, buffers.positions, buffers.velocities, {
    tolerance: options.constraintTolerance,
    velocities: true,
  });
  await phase(pool, "lfmiddle_drift", bindings, system.particleCount);
  await constrainOnDevice(pool, tables, buffers.positions, buffers.trial, {
    tolerance: options.constraintTolerance,
  });
  await phase(pool, "lfmiddle_finish", bindings, system.particleCount);
}

/** `ReduceParams`: a count and one over the total mass. */
function reduceParams(pool: BufferPool, system: OpenMMSystem): GPUBuffer {
  let totalMass = 0;
  for (let i = 0; i < system.particleCount; i++) {
    totalMass += system.masses[i] as number;
  }
  const bytes = new ArrayBuffer(16);
  new Uint32Array(bytes, 0, 1)[0] = system.particleCount;
  new Float32Array(bytes, 8, 1)[0] = totalMass === 0 ? 0 : 1 / totalMass;
  const buffer = pool.buffer({
    label: "reduce/params",
    byteLength: 16,
    usage: UNIFORM,
  });
  pool.device.queue.writeBuffer(buffer, 0, new Uint8Array(bytes));
  return buffer;
}

/** Zero the reduction accumulators. A fixed-point sum starts from exact zero. */
function clearTotals(pool: BufferPool, buffers: IntegratorBuffers): void {
  pool.device.queue.writeBuffer(buffers.totals, 0, new Uint32Array(6));
}

/**
 * `CMMotionRemover`: subtract the mass-weighted mean velocity.
 *
 * Two dispatches, because the second needs the first's total. They cannot be
 * merged without a device-wide barrier, which a compute pass does not offer.
 */
export async function removeCentreOfMassOnDevice(
  pool: BufferPool,
  system: OpenMMSystem,
  buffers: IntegratorBuffers,
): Promise<void> {
  const mass = pool.existing("system/mass");
  if (mass === undefined) {
    throw new Error("system/mass is not in the pool; call uploadConstraintTables first");
  }
  const bindings = [
    { binding: 0, resource: { buffer: buffers.velocities } },
    { binding: 1, resource: { buffer: mass } },
    { binding: 2, resource: { buffer: buffers.totals } },
    { binding: 3, resource: { buffer: reduceParams(pool, system) } },
  ];
  clearTotals(pool, buffers);
  const workgroups: readonly [number] = [Math.ceil(system.particleCount / 64)];
  await dispatch(pool.device, {
    label: "reduce/momentum",
    code: REDUCE,
    entryPoint: "momentum",
    bindings,
    bindGroupLayout: layoutsFor(pool.device).reduce,
    workgroups,
  });
  await dispatch(pool.device, {
    label: "reduce/subtract_com",
    code: REDUCE,
    entryPoint: "subtract_com",
    bindings,
    bindGroupLayout: layoutsFor(pool.device).reduce,
    workgroups,
  });
}

/**
 * The kinetic energy, kJ/mol, in the unshifted convention.
 *
 * That is what `LangevinIntegrator` reports and so what the notebook's log
 * shows. Verlet's conserved quantity needs velocities shifted half a step and
 * re-projected -- see src/reference/kinetic.ts and docs/DYNAMICS.md.
 */
export async function kineticEnergyOnDevice(
  pool: BufferPool,
  system: OpenMMSystem,
  buffers: IntegratorBuffers,
): Promise<number> {
  const mass = pool.existing("system/mass");
  if (mass === undefined) {
    throw new Error("system/mass is not in the pool; call uploadConstraintTables first");
  }
  clearTotals(pool, buffers);
  await dispatch(pool.device, {
    label: "reduce/kinetic",
    code: REDUCE,
    entryPoint: "kinetic",
    bindGroupLayout: layoutsFor(pool.device).reduce,
    bindings: [
      { binding: 0, resource: { buffer: buffers.velocities } },
      { binding: 1, resource: { buffer: mass } },
      { binding: 2, resource: { buffer: buffers.totals } },
      { binding: 3, resource: { buffer: reduceParams(pool, system) } },
    ],
    workgroups: [Math.ceil(system.particleCount / 64)],
  });
  const words = new Uint32Array(await readBuffer(pool.device, buffers.totals, 8));
  // The high word is signed; `| 0` is what reinterprets it as such.
  return fromFixed(fromWords(words[0] as number, (words[1] as number) | 0));
}

/** The mass-weighted momentum the last reduction measured, amu nm/ps. */
export async function readMomentum(
  pool: BufferPool,
  buffers: IntegratorBuffers,
): Promise<readonly [number, number, number]> {
  const words = new Uint32Array(await readBuffer(pool.device, buffers.totals, 24));
  const axis = (slot: number): number =>
    fromFixed(fromWords(words[slot * 2] as number, (words[slot * 2 + 1] as number) | 0));
  return [axis(0), axis(1), axis(2)];
}
