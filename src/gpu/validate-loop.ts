/**
 * The step loop, end to end.
 *
 * The force kernels, the constraints and the integrators each have their own
 * differential test already. What those cannot cover is the loop's own job:
 * that the passes are encoded in the right order, that the cell list is rebuilt
 * often enough, that nothing is read at a point where it has not been written,
 * and that a hundred steps in one submission behave like a hundred steps.
 *
 * Four checks, chosen so that each has one explanation if it fails.
 *
 * **Determinism.** Two runs of the same length from the same state must agree
 * bit for bit. This is the invariant in `AGENTS.md`, and it is the one a loop
 * can break in ways a single step cannot -- a stale cell list, a race between
 * passes, a parameter block read after it was overwritten. Asserted as exactly
 * zero, because anything else is a race.
 *
 * **One step against the reference.** The loop's own forces are read back and
 * handed to the f64 reference, which then takes the same step on the host. Any
 * disagreement is the loop's orchestration, not the force field: the two sides
 * are given identical forces by construction. This is the check that localises
 * an ordering mistake.
 *
 * **Energy conservation.** Velocity Verlet with no thermostat, over enough
 * steps for a drift to show. This is Phase 1's gate and it is sensitive to
 * things nothing else here is: a cell list left too long, a constraint
 * tolerance that is too loose, a position format that rounds one way.
 *
 * **Throughput.** Not a correctness check, but it belongs next to one: a number
 * measured on the same configuration the checks just passed on, so the
 * performance claim and the correctness claim are about the same code.
 */

import { parseState, type OpenMMState } from "../engine/state.js";
import { parseSystem, type OpenMMSystem } from "../engine/system.js";
import { splitConstraints } from "../engine/constraint-clusters.js";
import { verletStep, type IntegratorState } from "../reference/integrator.js";
import {
  centreOfMassRemover,
  removeCentreOfMassMotion,
  removesOnStep,
} from "../reference/com-motion.js";
import { unpackFixed } from "./forces.js";
import { readBuffer } from "./compute.js";
import { StepLoop, type Report, type RunOptions } from "./loop.js";

const TIME_STEP = 0.002;
const TOLERANCE = 1e-6;

/**
 * Acceptance thresholds.
 *
 * The one-step bound is the same 4e-7 nm the integrator kernels are held to,
 * because with identical forces that is the same comparison.
 *
 * The drift bound is calibrated rather than chosen. `tools/measure_drift.py`
 * runs OpenMM from the same captured configuration, with the same integrator
 * and timestep, on its CPU platform -- single-precision forces, double
 * accumulation, the closest comparison available without a GPU OpenCL stack --
 * and fits a least-squares slope:
 *
 *   OpenMM CPU   1L2Y 0.374   3HTB 0.385   kJ/mol per atom per ns
 *   this engine  1L2Y 0.282   3HTB 0.540
 *
 * Both from a least-squares slope over ten reports, on both sides. 1L2Y is
 * better than OpenMM's own number and 3HTB is within 40% of it; the gap between
 * the two fixtures closed when the minimum image moved into fixed point, which
 * took 3HTB's direct force error from 1.13e-5 to 5.37e-7 (docs/NONBONDED.md).
 *
 * So the gate is 1.5, a little under three times the worse measurement: a
 * regression test rather than a claim that either fixture is at its floor. The
 * real target is OpenMM's *mixed-precision GPU* platform, which has to be
 * measured on Colab -- this machine has no working OpenCL.
 */
export const LOOP_THRESHOLDS = {
  onePosition: 4e-7,
  oneVelocity: 4e-4,
  driftPerAtomPerNs: 1.5,
} as const;

export interface LoopComparison {
  readonly particleCount: number;
  readonly passesPerStep: number;
  /** nm. Must be exactly zero. */
  readonly determinismDifference: number;
  readonly onePositionVsReference: number;
  readonly oneVelocityVsReference: number;
  /** Steps the conservation run covered, and its span in ps. */
  readonly conservationSteps: number;
  readonly conservationPicoseconds: number;
  /** kJ/mol, between the first and last report. */
  readonly totalEnergyChange: number;
  /**
   * kJ/mol per atom per ns, from a least-squares slope over every report.
   *
   * 🔴 Not the endpoint difference, which is what this used to be and which is
   * not a drift at all: over a picosecond the total energy fluctuates by about
   * as much as it trends, so two endpoints cannot tell one from the other.
   * Measured, the endpoint figure for 1L2Y moved from 0.043 to 0.426 when the
   * forces got *more* accurate -- which is the signature of reading noise, not
   * of a regression. `tools/measure_drift.py` fits a slope for exactly this
   * reason and this had not been brought into line with it.
   */
  readonly driftPerAtomPerNs: number;
  /** The reports the slope was fitted to. */
  readonly reports: readonly Report[];
  readonly firstReport: Report;
  readonly lastReport: Report;
  /** ns/day at this timestep, measured over the conservation run. */
  readonly nsPerDay: number;
  readonly peakPoolBytes: number;
  readonly thresholds: typeof LOOP_THRESHOLDS;
  readonly ok: boolean;
}

/** A state whose coordinates come from a captured frame rather than the XML. */
function stateAtFrame(
  base: OpenMMState,
  positions: Float64Array,
  velocities: Float64Array,
): OpenMMState {
  return { ...base, positions, velocities };
}

const worst = (a: Float64Array, b: Float64Array): number => {
  let value = 0;
  for (let i = 0; i < a.length; i++) {
    value = Math.max(value, Math.abs((a[i] as number) - (b[i] as number)));
  }
  return value;
};

async function runFrom(
  device: GPUDevice,
  system: OpenMMSystem,
  state: OpenMMState,
  options: RunOptions,
  steps: number,
): Promise<{ positions: Float64Array; velocities: Float64Array }> {
  const loop = await StepLoop.fromParsed(device, system, state, options);
  try {
    await loop.run(steps);
    return {
      positions: await loop.positions(),
      velocities: await loop.velocities(),
    };
  } finally {
    loop.destroy();
  }
}

export async function validateLoop(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
  start: { readonly positions: Float64Array; readonly velocities: Float64Array },
  conservationSteps = 500,
): Promise<LoopComparison> {
  const system = parseSystem(texts.systemXml);
  const base = parseState(texts.stateXml);
  const state = stateAtFrame(base, start.positions, start.velocities);
  const verlet: RunOptions = {
    timeStep: TIME_STEP,
    temperature: 300,
    friction: 0,
    constraintTolerance: TOLERANCE,
    integrator: "verlet",
    // Every step: this list has no skin, and a stale one loses pairs. See
  };

  // 1. Determinism: the same run twice.
  const first = await runFrom(device, system, state, verlet, 25);
  const second = await runFrom(device, system, state, verlet, 25);
  const determinismDifference = Math.max(
    worst(first.positions, second.positions),
    worst(first.velocities, second.velocities),
  );

  // 2. One step, against the reference given the loop's own forces.
  const loop = await StepLoop.fromParsed(device, system, state, verlet);
  let onePositionVsReference = 0;
  let oneVelocityVsReference = 0;
  try {
    await loop.evaluateForces();
    const components = system.particleCount * 3;
    const forceWords = new Uint32Array(
      await readBuffer(device, loop.forceBuffer, components * 8),
    );
    const forces = unpackFixed(forceWords, components);
    const before: IntegratorState = {
      positions: await loop.positions(),
      velocities: await loop.velocities(),
    };
    await loop.run(1);
    const after = {
      positions: await loop.positions(),
      velocities: await loop.velocities(),
    };

    // The loop removes centre-of-mass motion before the forces, and the forces
    // do not depend on velocities, so the ones read above are the right ones.
    const remover = centreOfMassRemover(system);
    if (remover !== undefined && removesOnStep(remover.frequency, 0)) {
      removeCentreOfMassMotion(system, before.velocities);
    }
    verletStep(system, before, forces, TIME_STEP, TOLERANCE, splitConstraints(system));
    onePositionVsReference = worst(after.positions, before.positions);
    oneVelocityVsReference = worst(after.velocities, before.velocities);
  } finally {
    loop.destroy();
  }

  // 3 and 4. Conservation and throughput, on one run.
  const conserve = await StepLoop.fromParsed(device, system, state, verlet);
  const reports: Report[] = [];
  let nsPerDay = 0;
  let peakPoolBytes = 0;
  let passesPerStep = 0;
  // Ten intervals, so a slope has something to be fitted to. Each report
  // re-evaluates the forces at the current coordinates, which is why it is ten
  // and not a hundred.
  const interval = Math.max(1, Math.floor(conservationSteps / 10));
  try {
    passesPerStep = conserve.passesPerStep;
    reports.push(await conserve.report());
    const began = performance.now();
    let done = 0;
    while (done < conservationSteps) {
      const chunk = Math.min(interval, conservationSteps - done);
      await conserve.run(chunk);
      done += chunk;
      reports.push(await conserve.report());
    }
    const elapsed = (performance.now() - began) / 1000;
    const nanoseconds = (conservationSteps * TIME_STEP) / 1000;
    // The reports are inside the timing, so this is a run that reports ten
    // times -- which is a realistic production cadence, not a bare loop.
    nsPerDay = elapsed === 0 ? 0 : (nanoseconds / elapsed) * 86400;
    peakPoolBytes = conserve.report_pool().peakBytes;
  } finally {
    conserve.destroy();
  }

  const firstReport = reports[0] as Report;
  const lastReport = reports[reports.length - 1] as Report;
  const totalEnergyChange = lastReport.totalEnergy - firstReport.totalEnergy;
  // Least squares over (time in ns, total energy).
  let sumT = 0;
  let sumE = 0;
  let sumTT = 0;
  let sumTE = 0;
  for (const report of reports) {
    const t = report.time / 1000;
    sumT += t;
    sumE += report.totalEnergy;
    sumTT += t * t;
    sumTE += t * report.totalEnergy;
  }
  const n = reports.length;
  const denominator = n * sumTT - sumT * sumT;
  const slope = denominator === 0 ? 0 : (n * sumTE - sumT * sumE) / denominator;
  const driftPerAtomPerNs = Math.abs(slope / system.particleCount);

  return {
    particleCount: system.particleCount,
    passesPerStep,
    determinismDifference,
    onePositionVsReference,
    oneVelocityVsReference,
    conservationSteps,
    conservationPicoseconds: conservationSteps * TIME_STEP,
    totalEnergyChange,
    driftPerAtomPerNs,
    reports,
    firstReport,
    lastReport,
    nsPerDay,
    peakPoolBytes,
    thresholds: LOOP_THRESHOLDS,
    ok:
      determinismDifference === 0 &&
      onePositionVsReference < LOOP_THRESHOLDS.onePosition &&
      oneVelocityVsReference < LOOP_THRESHOLDS.oneVelocity &&
      driftPerAtomPerNs < LOOP_THRESHOLDS.driftPerAtomPerNs,
  };
}
