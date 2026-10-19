/**
 * The integrator kernels against the reference *and* against OpenMM.
 *
 * Driven by the forces OpenMM recorded at each captured frame rather than by
 * our own force evaluation, which is what isolates the integrator: a
 * disagreement here cannot be a force-field error leaking in. The reference
 * tests use the same fixtures and reach 5.7e-9 nm a step, so OpenMM's frames
 * are a usable target for the device too, and both numbers are reported --
 * agreement with the reference says the kernel is a faithful transcription,
 * agreement with OpenMM says the reference was right to begin with.
 *
 * Every step restarts from OpenMM's own frame, so errors cannot accumulate and
 * each step is an independent measurement. That also means the residual the
 * drift carries is zeroed each time: in a run it persists by design, but here
 * it would be comparing a device that remembers against a reference that does
 * not.
 *
 * Four things are checked, in the order they fail:
 *
 *   1. both integrators, step by step, against the reference and OpenMM;
 *   2. the thermostat with friction on, which only works because the generator
 *      is counter-based and the host can draw the same deviates;
 *   3. `CMMotionRemover`, whose reduction is exact and so must match to the
 *      precision of the terms, not of the sum;
 *   4. the reported kinetic energy.
 */

import { parseState } from "../engine/state.js";
import { parseSystem, type OpenMMSystem } from "../engine/system.js";
import { splitConstraints } from "../engine/constraint-clusters.js";
import {
  langevinMiddleStep,
  verletStep,
  type IntegratorState,
} from "../reference/integrator.js";
import {
  centreOfMassRemover,
  removeCentreOfMassMotion,
  removesOnStep,
} from "../reference/com-motion.js";
import { kineticEnergy } from "../reference/kinetic.js";
import { thermostatNormals } from "../reference/random.js";
import { constrainPositions, constrainVelocities } from "../reference/constraints.js";
import { BufferPool } from "./allocator.js";
import {
  constrainOnDevice,
  packConstraints,
  uploadConstraintTables,
} from "./constrain.js";
import { packPositions, unpackFixed } from "./forces.js";
import { readBuffer } from "./compute.js";
import {
  dispatchPhase,
  kineticEnergyOnDevice,
  langevinStepOnDevice,
  prepareIntegrator,
  readMomentum,
  removeCentreOfMassOnDevice,
  verletStepOnDevice,
  type IntegratorBuffers,
} from "./integrate.js";

const TIME_STEP = 0.002;
const TOLERANCE = 1e-6;
const TEMPERATURE = 300;
const SEED = 0x5eed;

/**
 * Acceptance thresholds for a single step.
 *
 * The position bound is set by the constraint kernels, which sit 8.7e-8 nm from
 * the reference on their own (src/gpu/validate-constraints.ts); nothing the
 * integrator adds is near that. The velocity bound is that number divided by
 * the timestep, because the constrained displacement *is* the velocity: a
 * 1e-7 nm position difference over 2 fs is 5e-5 nm/ps, and no tighter bound is
 * available without a tighter constraint solve.
 *
 * The energy bound is relative and loose by comparison, because a kinetic
 * energy is a sum of 3N squares of f32 velocities: the terms carry 6e-8 each
 * and the sum is exact, so the total lands around 1e-7 relative.
 */
export const INTEGRATOR_THRESHOLDS = {
  position: 4e-7,
  velocity: 4e-4,
  energyRelative: 1e-5,
  /** nm/ps, on the centre-of-mass velocity a reduction reports. */
  comVelocity: 1e-7,
} as const;

export interface TrajectoryFrames {
  readonly steps: number;
  /** (steps + 1) x 3N, nm. */
  readonly positions: Float64Array;
  /** (steps + 1) x 3N, nm/ps. */
  readonly velocities: Float64Array;
  /** (steps + 1) x 3N, kJ/mol/nm. */
  readonly forces: Float64Array;
}

export interface StepComparison {
  readonly integrator: "verlet" | "lfmiddle";
  readonly steps: number;
  readonly worstPositionVsReference: number;
  readonly worstVelocityVsReference: number;
  readonly worstPositionVsOpenmm: number;
  readonly worstVelocityVsOpenmm: number;
  readonly ok: boolean;
}

/**
 * Where a thermostatted step first disagrees.
 *
 * The step is five dispatches with two constraint solves between them, and the
 * stages amplify each other: a velocity difference becomes a position
 * difference over the drift, and `lfmiddle_finish` divides a position
 * difference by the timestep, multiplying it by 500. So a single end-to-end
 * number cannot say where the error entered, and the stage that *introduces*
 * it is the only one worth looking at.
 */
export interface ThermostatStages {
  readonly afterKick: number;
  readonly afterRattle: number;
  readonly afterDriftVelocity: number;
  readonly afterDriftPosition: number;
  readonly afterShake: number;
  readonly afterFinishVelocity: number;
}

export interface ThermostatComparison {
  readonly friction: number;
  readonly stages: ThermostatStages;
  readonly worstPositionVsReference: number;
  readonly worstVelocityVsReference: number;
  /** How far the velocities moved, so the comparison above has a scale. */
  readonly velocityChange: number;
  readonly ok: boolean;
}

export interface ReductionComparison {
  /** nm/ps, the centre-of-mass velocity the device removed. */
  readonly comVelocity: readonly [number, number, number];
  readonly comVelocityError: number;
  readonly worstVelocityVsReference: number;
  readonly kineticEnergy: number;
  readonly kineticEnergyRelativeError: number;
  readonly ok: boolean;
}

export interface IntegratorReport {
  readonly fixture: string;
  readonly particleCount: number;
  readonly steps: readonly StepComparison[];
  readonly thermostat: ThermostatComparison;
  readonly reductions: ReductionComparison;
  readonly thresholds: typeof INTEGRATOR_THRESHOLDS;
  readonly ok: boolean;
}

/** The worst per-component difference between two arrays. */
function worst(a: Float64Array, b: Float64Array): number {
  let value = 0;
  for (let i = 0; i < a.length; i++) {
    value = Math.max(value, Math.abs((a[i] as number) - (b[i] as number)));
  }
  return value;
}

const relative = (got: number, want: number): number =>
  want === 0 ? Math.abs(got) : Math.abs(got - want) / Math.abs(want);

/** Put a host state onto the device, residual cleared. */
function upload(
  pool: BufferPool,
  buffers: IntegratorBuffers,
  state: { positions: Float64Array; velocities: Float64Array },
  forces?: Float64Array,
): void {
  const queue = pool.device.queue;
  const put = (buffer: GPUBuffer, data: ArrayBufferView): void =>
    queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
  put(buffers.positions, packPositions(state.positions));
  put(buffers.velocities, Float32Array.from(state.velocities));
  put(buffers.residual, new Float32Array(state.velocities.length));
  if (forces !== undefined) put(buffers.forces, packPositions(forces));
}

/** Read the device's state back as doubles. */
async function download(
  pool: BufferPool,
  buffers: IntegratorBuffers,
  components: number,
): Promise<{ positions: Float64Array; velocities: Float64Array }> {
  const positionWords = new Uint32Array(
    await readBuffer(pool.device, buffers.positions, components * 8),
  );
  const velocityBytes = new Float32Array(
    await readBuffer(pool.device, buffers.velocities, components * 4),
  );
  return {
    positions: unpackFixed(positionWords, components),
    velocities: Float64Array.from(velocityBytes),
  };
}

async function compareSteps(
  pool: BufferPool,
  system: OpenMMSystem,
  tables: ReturnType<typeof packConstraints>,
  buffers: IntegratorBuffers,
  frames: TrajectoryFrames,
  integrator: "verlet" | "lfmiddle",
): Promise<StepComparison> {
  const components = system.particleCount * 3;
  const split = splitConstraints(system);
  const remover = centreOfMassRemover(system);
  const frame = (array: Float64Array, index: number): Float64Array =>
    array.subarray(index * components, (index + 1) * components);

  let worstPositionVsReference = 0;
  let worstVelocityVsReference = 0;
  let worstPositionVsOpenmm = 0;
  let worstVelocityVsOpenmm = 0;

  for (let step = 0; step < frames.steps; step++) {
    const start = {
      positions: Float64Array.from(frame(frames.positions, step)),
      velocities: Float64Array.from(frame(frames.velocities, step)),
    };
    const forces = frame(frames.forces, step);

    // `updateContextState` runs before the forces, and that is where the
    // centre-of-mass remover acts. Our fixtures carry one at frequency 1.
    if (remover !== undefined && removesOnStep(remover.frequency, step)) {
      removeCentreOfMassMotion(system, start.velocities);
    }

    upload(pool, buffers, start, forces);
    const options = {
      timeStep: TIME_STEP,
      constraintTolerance: TOLERANCE,
      temperature: TEMPERATURE,
      friction: 0,
      step,
    };
    if (integrator === "verlet") {
      await verletStepOnDevice(pool, system, tables, buffers, options);
    } else {
      await langevinStepOnDevice(pool, system, tables, buffers, options);
    }
    const device = await download(pool, buffers, components);

    const reference: IntegratorState = {
      positions: Float64Array.from(start.positions),
      velocities: Float64Array.from(start.velocities),
    };
    if (integrator === "verlet") {
      verletStep(system, reference, forces, TIME_STEP, TOLERANCE, split);
    } else {
      langevinMiddleStep(system, reference, forces, {
        timeStep: TIME_STEP,
        temperature: TEMPERATURE,
        friction: 0,
        constraintTolerance: TOLERANCE,
        split,
      });
    }

    worstPositionVsReference = Math.max(
      worstPositionVsReference,
      worst(device.positions, reference.positions),
    );
    worstVelocityVsReference = Math.max(
      worstVelocityVsReference,
      worst(device.velocities, reference.velocities),
    );
    worstPositionVsOpenmm = Math.max(
      worstPositionVsOpenmm,
      worst(device.positions, frame(frames.positions, step + 1)),
    );
    worstVelocityVsOpenmm = Math.max(
      worstVelocityVsOpenmm,
      worst(device.velocities, frame(frames.velocities, step + 1)),
    );
  }

  return {
    integrator,
    steps: frames.steps,
    worstPositionVsReference,
    worstVelocityVsReference,
    worstPositionVsOpenmm,
    worstVelocityVsOpenmm,
    ok:
      worstPositionVsReference < INTEGRATOR_THRESHOLDS.position &&
      worstVelocityVsReference < INTEGRATOR_THRESHOLDS.velocity &&
      worstPositionVsOpenmm < INTEGRATOR_THRESHOLDS.position &&
      worstVelocityVsOpenmm < INTEGRATOR_THRESHOLDS.velocity,
  };
}

/**
 * The thermostat, with friction switched on.
 *
 * This is the check a sequential generator makes impossible. Both sides draw
 * the same deviates from the same (particle, axis, step) counters, so the noise
 * is not a statistical nuisance to be averaged away -- it is just another term
 * that has to agree. `velocityChange` is reported alongside so the difference
 * can be read against how much the thermostat actually did.
 */
async function compareThermostat(
  pool: BufferPool,
  system: OpenMMSystem,
  tables: ReturnType<typeof packConstraints>,
  buffers: IntegratorBuffers,
  frames: TrajectoryFrames,
): Promise<ThermostatComparison> {
  const components = system.particleCount * 3;
  const split = splitConstraints(system);
  const friction = 1;
  const start = {
    positions: Float64Array.from(frames.positions.subarray(0, components)),
    velocities: Float64Array.from(frames.velocities.subarray(0, components)),
  };
  const forces = frames.forces.subarray(0, components);

  upload(pool, buffers, start, forces);
  await langevinStepOnDevice(pool, system, tables, buffers, {
    timeStep: TIME_STEP,
    constraintTolerance: TOLERANCE,
    temperature: TEMPERATURE,
    friction,
    seed: SEED,
    step: 0,
  });
  const device = await download(pool, buffers, components);

  const reference: IntegratorState = {
    positions: Float64Array.from(start.positions),
    velocities: Float64Array.from(start.velocities),
  };
  langevinMiddleStep(system, reference, forces, {
    timeStep: TIME_STEP,
    temperature: TEMPERATURE,
    friction,
    constraintTolerance: TOLERANCE,
    normals: thermostatNormals(SEED, 0),
    split,
  });

  const worstPositionVsReference = worst(device.positions, reference.positions);
  const worstVelocityVsReference = worst(device.velocities, reference.velocities);
  const stages = await stageThermostat(pool, system, tables, buffers, start, forces, friction);
  return {
    friction,
    stages,
    worstPositionVsReference,
    worstVelocityVsReference,
    velocityChange: worst(reference.velocities, start.velocities),
    ok:
      worstPositionVsReference < INTEGRATOR_THRESHOLDS.position &&
      worstVelocityVsReference < INTEGRATOR_THRESHOLDS.velocity,
  };
}

/**
 * Re-run the same step one dispatch at a time, comparing each stage.
 *
 * The reference side is assembled here from `constrainPositions`,
 * `constrainVelocities` and the scheme's arithmetic rather than by calling
 * `langevinMiddleStep`, because the point is to see inside it. The arithmetic
 * is duplicated, which is a cost worth paying exactly once: src/reference is
 * still the only definition of the step, and test/integrator.test.ts holds that
 * definition to OpenMM's own trajectory.
 */
async function stageThermostat(
  pool: BufferPool,
  system: OpenMMSystem,
  tables: ReturnType<typeof packConstraints>,
  buffers: IntegratorBuffers,
  start: { positions: Float64Array; velocities: Float64Array },
  forces: Float64Array,
  friction: number,
): Promise<ThermostatStages> {
  const components = system.particleCount * 3;
  const split = splitConstraints(system);
  const inverse = Float64Array.from(system.masses, (m) => (m === 0 ? 0 : 1 / m));
  const kT = 0.008314462618153243 * TEMPERATURE;
  const velocityScale = Math.exp(-TIME_STEP * friction);
  const noiseScale = Math.sqrt(1 - velocityScale * velocityScale);
  const draw = thermostatNormals(SEED, 0);
  const halfStep = 0.5 * TIME_STEP;

  const options = {
    timeStep: TIME_STEP,
    constraintTolerance: TOLERANCE,
    temperature: TEMPERATURE,
    friction,
    seed: SEED,
    step: 0,
  };

  const positions = Float64Array.from(start.positions);
  const velocities = Float64Array.from(start.velocities);
  const readVelocities = async (): Promise<Float64Array> =>
    Float64Array.from(
      new Float32Array(await readBuffer(pool.device, buffers.velocities, components * 4)),
    );
  const readFixed = async (buffer: GPUBuffer): Promise<Float64Array> =>
    unpackFixed(
      new Uint32Array(await readBuffer(pool.device, buffer, components * 8)),
      components,
    );

  upload(pool, buffers, start, forces);

  await dispatchPhase(pool, "lfmiddle_kick", buffers, system, options);
  for (let i = 0; i < system.particleCount; i++) {
    if ((inverse[i] as number) === 0) continue;
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      velocities[at] =
        (velocities[at] as number) +
        TIME_STEP * (inverse[i] as number) * (forces[at] as number);
    }
  }
  const afterKick = worst(await readVelocities(), velocities);

  await constrainOnDevice(pool, tables, buffers.positions, buffers.velocities, {
    tolerance: TOLERANCE,
    velocities: true,
  });
  constrainVelocities(system, positions, velocities, TOLERANCE, split);
  const afterRattle = worst(await readVelocities(), velocities);

  await dispatchPhase(pool, "lfmiddle_drift", buffers, system, options);
  const trial = Float64Array.from(positions);
  for (let i = 0; i < system.particleCount; i++) {
    const inverseMass = inverse[i] as number;
    if (inverseMass === 0) continue;
    const noise = noiseScale * Math.sqrt(kT * inverseMass);
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      trial[at] = (trial[at] as number) + (velocities[at] as number) * halfStep;
      velocities[at] =
        velocityScale * (velocities[at] as number) +
        (noise === 0 ? 0 : noise * draw(i, axis));
      trial[at] = (trial[at] as number) + (velocities[at] as number) * halfStep;
    }
  }
  const afterDriftVelocity = worst(await readVelocities(), velocities);
  const afterDriftPosition = worst(await readFixed(buffers.trial), trial);

  const unconstrained = Float64Array.from(trial);
  await constrainOnDevice(pool, tables, buffers.positions, buffers.trial, {
    tolerance: TOLERANCE,
  });
  constrainPositions(system, positions, trial, TOLERANCE, split);
  const afterShake = worst(await readFixed(buffers.trial), trial);

  await dispatchPhase(pool, "lfmiddle_finish", buffers, system, options);
  const scale = 1 / TIME_STEP;
  for (let i = 0; i < system.particleCount; i++) {
    if ((inverse[i] as number) === 0) continue;
    for (let axis = 0; axis < 3; axis++) {
      const at = i * 3 + axis;
      velocities[at] =
        (velocities[at] as number) +
        scale * ((trial[at] as number) - (unconstrained[at] as number));
    }
  }
  const afterFinishVelocity = worst(await readVelocities(), velocities);

  return {
    afterKick,
    afterRattle,
    afterDriftVelocity,
    afterDriftPosition,
    afterShake,
    afterFinishVelocity,
  };
}

async function compareReductions(
  pool: BufferPool,
  system: OpenMMSystem,
  buffers: IntegratorBuffers,
  frames: TrajectoryFrames,
): Promise<ReductionComparison> {
  const components = system.particleCount * 3;
  const start = {
    positions: Float64Array.from(frames.positions.subarray(0, components)),
    velocities: Float64Array.from(frames.velocities.subarray(0, components)),
  };

  upload(pool, buffers, start);
  const expectedKinetic = kineticEnergy(system, start.velocities);
  const measuredKinetic = await kineticEnergyOnDevice(pool, system, buffers);

  await removeCentreOfMassOnDevice(pool, system, buffers);
  const momentum = await readMomentum(pool, buffers);
  const device = await download(pool, buffers, components);

  let totalMass = 0;
  for (let i = 0; i < system.particleCount; i++) {
    totalMass += system.masses[i] as number;
  }
  const comVelocity: readonly [number, number, number] = [
    momentum[0] / totalMass,
    momentum[1] / totalMass,
    momentum[2] / totalMass,
  ];

  const reference = Float64Array.from(start.velocities);
  const expectedCom = removeCentreOfMassMotion(system, reference);
  const comVelocityError = Math.max(
    ...[0, 1, 2].map((axis) =>
      Math.abs((comVelocity[axis] as number) - (expectedCom[axis] as number)),
    ),
  );

  const worstVelocityVsReference = worst(device.velocities, reference);
  const kineticEnergyRelativeError = relative(measuredKinetic, expectedKinetic);
  return {
    comVelocity,
    comVelocityError,
    worstVelocityVsReference,
    kineticEnergy: measuredKinetic,
    kineticEnergyRelativeError,
    ok:
      comVelocityError < INTEGRATOR_THRESHOLDS.comVelocity &&
      worstVelocityVsReference < INTEGRATOR_THRESHOLDS.velocity &&
      kineticEnergyRelativeError < INTEGRATOR_THRESHOLDS.energyRelative,
  };
}

/** Run every integrator check on one fixture. */
export async function validateIntegrators(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
  trajectories: {
    readonly verlet: TrajectoryFrames;
    readonly lfmiddle: TrajectoryFrames;
  },
  fixtureName: string,
): Promise<IntegratorReport> {
  const system = parseSystem(texts.systemXml);
  parseState(texts.stateXml);
  const split = splitConstraints(system);
  const tables = packConstraints(system, split);
  const pool = new BufferPool(device);
  try {
    uploadConstraintTables(pool, tables);
    const buffers = prepareIntegrator(pool, system);
    const steps = [
      await compareSteps(pool, system, tables, buffers, trajectories.verlet, "verlet"),
      await compareSteps(
        pool,
        system,
        tables,
        buffers,
        trajectories.lfmiddle,
        "lfmiddle",
      ),
    ];
    const thermostat = await compareThermostat(
      pool,
      system,
      tables,
      buffers,
      trajectories.lfmiddle,
    );
    const reductions = await compareReductions(
      pool,
      system,
      buffers,
      trajectories.lfmiddle,
    );
    return {
      fixture: fixtureName,
      particleCount: system.particleCount,
      steps,
      thermostat,
      reductions,
      thresholds: INTEGRATOR_THRESHOLDS,
      ok: steps.every((s) => s.ok) && thermostat.ok && reductions.ok,
    };
  } finally {
    pool.destroy();
  }
}
