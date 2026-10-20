/**
 * The run: minimise, equilibrate, produce by strides.
 *
 * This is the notebook's protocol as a state machine, and it is deliberately
 * independent of where the dynamics happen. It talks to a `ProtocolEngine` --
 * which `src/gpu/loop.ts` implements -- and to a `RunOutputs` sink, so the
 * orchestration can be driven in Node against a stub and in a browser against
 * the real thing. The orchestration is where the mistakes live: interval
 * arithmetic, the step and time counters that have to continue across strides,
 * and which stages a resumed run is allowed to skip.
 *
 * The stages, from docs/NOTEBOOK_PARITY.md:
 *
 *   **Equilibration**, NPT. Restraints are added *first*, so the minimisation
 *   is restrained -- finding 10, and easy to get backwards. Then minimise, then
 *   `setVelocitiesToTemperature`, then dynamics with LFMiddle and the barostat.
 *
 *   **Production**, NPT, by strides. A *fresh* system with no restraints, the
 *   state loaded from equilibration's checkpoint, and one trajectory, log,
 *   checkpoint and final frame per stride. A stride whose checkpoint already
 *   exists is skipped, which is how a disconnected Colab session resumes and is
 *   the model this keeps.
 */

import { serialiseState, type OpenMMState } from "./state.js";
import type { OpenMMSystem, Vec3 } from "./system.js";
import { DcdWriter } from "../analysis/dcd.js";
import { rewritePdb } from "../analysis/pdb.js";
import { logHeader, logRow } from "../analysis/log.js";

/** What the protocol needs from whatever is integrating. */
export interface ProtocolEngine {
  readonly particleCount: number;
  /** Steps run so far, which the protocol does not reset between stages. */
  readonly steps: number;
  positions(): Promise<Float64Array>;
  velocities(): Promise<Float64Array>;
  boxLengths(): Vec3;
  setPositions(positions: Float64Array): void;
  setVelocities(velocities: Float64Array): void;
  setBox(box: Vec3): void;
  /** Advance by `steps`. */
  run(steps: number): Promise<void>;
  /** Energies and temperature at the current configuration. */
  sample(): Promise<{
    readonly potentialEnergy: number;
    readonly kineticEnergy: number;
    readonly totalEnergy: number;
    readonly temperature: number;
  }>;
  /** Minimise in place, returning the energy reached. */
  minimise(maxIterations: number, tolerance: number): Promise<number>;
}

/**
 * Where artefacts go.
 *
 * Narrow on purpose: append, patch a range, write text, read text back. OPFS
 * gives all four, a Node filesystem gives all four, and an in-memory map gives
 * all four -- which is what makes the protocol testable. `read` returning
 * undefined is how resume is detected.
 */
export interface RunOutputs {
  append(name: string, bytes: Uint8Array): Promise<void>;
  patch(name: string, offset: number, bytes: Uint8Array): Promise<void>;
  write(name: string, text: string): Promise<void>;
  read(name: string): Promise<string | undefined>;
  exists(name: string): Promise<boolean>;
}

export interface StageSettings {
  /** ns. */
  readonly nanoseconds: number;
  /** ps. */
  readonly timeStep: number;
  /** K. */
  readonly temperature: number;
  /** bar. */
  readonly pressure: number;
  /** ps between trajectory frames. */
  readonly trajectoryInterval: number;
  /** ps between log rows. */
  readonly logInterval: number;
}

export interface ProtocolSettings {
  readonly jobName: string;
  readonly equilibration: StageSettings & {
    readonly minimisationSteps: number;
    /** kJ/mol/nm, the minimiser's tolerance. OpenMM's default is 10. */
    readonly minimisationTolerance?: number;
    /** kJ/mol/nm^2. Zero skips the restraint force entirely, as the notebook does. */
    readonly restraintForceConstant: number;
  };
  readonly production: StageSettings & {
    readonly strides: number;
  };
  /** A run records its seed. */
  readonly seed: number;
}

/**
 * How many steps an interval in picoseconds is.
 *
 * 🔴 Refuses a non-integer rather than rounding. A 10 ps interval at a 3 fs
 * timestep is 3333.33 steps, and silently rounding it makes every frame time in
 * the trajectory wrong by a growing amount -- which is invisible until someone
 * compares a DCD's frame count against the log's step numbers. The notebook's
 * own timestep options include 3 fs, so this is reachable.
 */
export function stepsForInterval(
  picoseconds: number,
  timeStep: number,
  what: string,
): number {
  const steps = picoseconds / timeStep;
  const rounded = Math.round(steps);
  if (Math.abs(steps - rounded) > 1e-9 || rounded < 1) {
    throw new Error(
      `${what} of ${picoseconds} ps is ${steps} steps at a ${timeStep} ps ` +
        "timestep, which is not a whole number. Choose an interval that " +
        "divides by the timestep: otherwise the frame times drift from the " +
        "step numbers by a growing amount, and nothing reports it.",
    );
  }
  return rounded;
}

/** The step count a stage of `nanoseconds` covers. */
export const stepsForStage = (nanoseconds: number, timeStep: number): number =>
  stepsForInterval(nanoseconds * 1000, timeStep, "a stage length");

export interface StageResult {
  readonly name: string;
  readonly steps: number;
  readonly frames: number;
  readonly logRows: number;
  /** Seconds of wall clock. */
  readonly elapsed: number;
  readonly nsPerDay: number;
  readonly skipped: boolean;
}

export interface RunRecord {
  readonly jobName: string;
  readonly seed: number;
  readonly settings: ProtocolSettings;
  readonly particleCount: number;
  readonly stages: readonly StageResult[];
  readonly minimisedEnergy: number | null;
  readonly artefacts: readonly string[];
  readonly startedAt: string;
  readonly finishedAt: string | null;
}

/** Artefact names a stage produces, from its basename. */
export const artefactNames = (
  base: string,
): {
  readonly trajectory: string;
  readonly log: string;
  readonly checkpoint: string;
  readonly finalFrame: string;
} => ({
  trajectory: `${base}.dcd`,
  log: `${base}.log`,
  checkpoint: `${base}.rst`,
  // The notebook's `Equilibrated_PDB`: a topology for the analysis stage, not a
  // restart point. A PDB holds 0.001 A, which is a thousand times coarser than
  // the constraint tolerance -- the `.rst` is the checkpoint.
  finalFrame: `${base}.pdb`,
});

export interface StageOptions {
  /** Skip the stage when its checkpoint already exists. */
  readonly resume?: boolean;
  /**
   * The uploaded PDB, for the final frame.
   *
   * Left out, no PDB is written: the engine models no topology of its own, so
   * there is nothing to generate one from, and inventing atom names would be
   * worse than omitting the file.
   */
  readonly topologyPdb?: string;
  /** Called after every report, for a live view. */
  readonly onReport?: (row: {
    readonly step: number;
    readonly totalSteps: number;
    readonly potentialEnergy: number;
    readonly temperature: number;
    readonly nsPerDay: number;
  }) => void;
}

/**
 * Run one stage, writing a trajectory, a log and a checkpoint.
 *
 * 🔴 Reports land on steps that are multiples of the interval measured on the
 * **absolute** step counter, not on the stage's own. That is what OpenMM does --
 * `describeNextReport` returns `interval - currentStep % interval` -- and it is
 * why the notebook assigns `simulation.currentStep` by hand when it starts a
 * stride: without it each stride would restart the cadence and the step numbers
 * in the logs would not line up with the frames in the trajectories. So nothing
 * is reported at step zero either, which is a difference a frame count will
 * show.
 */
export async function runStage(
  engine: ProtocolEngine,
  outputs: RunOutputs,
  base: string,
  settings: StageSettings,
  options: StageOptions = {},
): Promise<StageResult> {
  const names = artefactNames(base);
  if (options.resume === true && (await outputs.exists(names.checkpoint))) {
    // The notebook's resume: a stride whose `.rst` exists is already done.
    return {
      name: base,
      steps: 0,
      frames: 0,
      logRows: 0,
      elapsed: 0,
      nsPerDay: 0,
      skipped: true,
    };
  }

  const totalSteps = stepsForStage(settings.nanoseconds, settings.timeStep);
  const frameInterval = stepsForInterval(
    settings.trajectoryInterval,
    settings.timeStep,
    "a trajectory interval",
  );
  const logInterval = stepsForInterval(
    settings.logInterval,
    settings.timeStep,
    "a log interval",
  );

  const writer = new DcdWriter({
    particleCount: engine.particleCount,
    timeStep: settings.timeStep,
    // 🔴 The step of the **first frame**, not where the stage began. OpenMM's
    // `DCDReporter` constructs its file lazily, inside the first `report()`
    // call -- by which time `simulation.currentStep` is already the step the
    // first frame belongs to. Passing the stage's start instead leaves both
    // `ISTART` and the patched `NSTEP` one whole interval low, which a reader
    // turns into a time axis shifted by one frame.
    firstStep: engine.steps + frameInterval,
    interval: frameInterval,
    periodic: true,
    titles: [`Created by making-it-rain-webgpu: ${base}`, ""],
  });
  await outputs.append(names.trajectory, writer.header());
  await outputs.write(names.log, logHeader());

  const began = performance.now();
  const finalStep = engine.steps + totalSteps;
  let frames = 0;
  let logRows = 0;
  let log = logHeader();

  while (engine.steps < finalStep) {
    // The next step at which anything is due, on the absolute counter.
    const untilFrame = frameInterval - (engine.steps % frameInterval);
    const untilLog = logInterval - (engine.steps % logInterval);
    const chunk = Math.min(untilFrame, untilLog, finalStep - engine.steps);
    await engine.run(chunk);

    const elapsed = (performance.now() - began) / 1000;
    if (engine.steps % frameInterval === 0) {
      await outputs.append(
        names.trajectory,
        writer.frame(await engine.positions(), engine.boxLengths()),
      );
      frames++;
    }
    if (engine.steps % logInterval === 0) {
      const sample = await engine.sample();
      const box = engine.boxLengths();
      const volume =
        (box[0] as number) * (box[1] as number) * (box[2] as number);
      const done = engine.steps - (finalStep - totalSteps);
      const nanoseconds = (done * settings.timeStep) / 1000;
      const nsPerDay = elapsed > 0 ? (nanoseconds / elapsed) * 86400 : null;
      log += logRow({
        step: engine.steps,
        potentialEnergy: sample.potentialEnergy,
        kineticEnergy: sample.kineticEnergy,
        totalEnergy: sample.totalEnergy,
        temperature: sample.temperature,
        volume,
        speed: nsPerDay,
      });
      logRows++;
      options.onReport?.({
        step: engine.steps,
        totalSteps: finalStep,
        potentialEnergy: sample.potentialEnergy,
        temperature: sample.temperature,
        nsPerDay: nsPerDay ?? 0,
      });
      // Rewritten whole rather than appended, so a run killed mid-write leaves
      // a readable log rather than a truncated row.
      await outputs.write(names.log, log);
    }
  }

  // The frame count was not known when the header went out.
  for (const { offset, bytes } of writer.headerPatches()) {
    await outputs.patch(names.trajectory, offset, bytes);
  }

  const box = engine.boxLengths();
  const checkpoint: OpenMMState = {
    openmmVersion: "8.4",
    stepCount: engine.steps,
    time: engine.steps * settings.timeStep,
    boxVectors: [
      [box[0] as number, 0, 0],
      [0, box[1] as number, 0],
      [0, 0, box[2] as number],
    ],
    particleCount: engine.particleCount,
    positions: await engine.positions(),
    velocities: await engine.velocities(),
  };
  await outputs.write(names.checkpoint, serialiseState(checkpoint));

  if (options.topologyPdb !== undefined) {
    await outputs.write(
      names.finalFrame,
      rewritePdb(options.topologyPdb, checkpoint.positions, {
        box: engine.boxLengths(),
        remark: `FINAL FRAME OF ${base.toUpperCase()}, STEP ${engine.steps}`,
      }),
    );
  }

  const elapsed = (performance.now() - began) / 1000;
  const nanoseconds = (totalSteps * settings.timeStep) / 1000;
  return {
    name: base,
    steps: totalSteps,
    frames,
    logRows,
    elapsed,
    nsPerDay: elapsed > 0 ? (nanoseconds / elapsed) * 86400 : 0,
    skipped: false,
  };
}

/** Stage basenames, as the notebook names them. */
export const stageNames = (
  settings: ProtocolSettings,
): { readonly equilibration: string; readonly strides: readonly string[] } => ({
  equilibration: `${settings.jobName}_equil`,
  strides: Array.from(
    { length: settings.production.strides },
    // The notebook numbers strides from 1 and puts the number in the basename,
    // which is what makes a half-finished run resumable by inspection.
    (_unused, index) => `${settings.jobName}_prod_${index + 1}`,
  ),
});

export interface ProtocolHooks {
  /** The uploaded PDB, written as each stage's final frame. */
  readonly topologyPdb?: string;
  /**
   * Prepare the engine for a stage.
   *
   * Equilibration runs with restraints and production without, and the notebook
   * builds a *fresh* `System` for production rather than removing the force.
   * The protocol cannot do that itself -- it does not own the engine -- so it
   * asks, and the caller returns whichever engine the stage needs.
   */
  readonly engineFor: (
    stage: "equilibration" | "production",
    state: OpenMMState | null,
  ) => Promise<ProtocolEngine>;
  /** Draw initial velocities, after minimisation and before dynamics. */
  readonly initialVelocities: (engine: ProtocolEngine) => Promise<Float64Array>;
  readonly onStage?: (result: StageResult) => void;
  readonly onReport?: StageOptions["onReport"];
}

/**
 * The whole protocol.
 *
 * Returns the record that becomes `run.json`: every setting, the seed, what each
 * stage did, and which artefacts exist. A resumed run reports the stages it
 * skipped rather than omitting them, so the record describes the trajectory on
 * disk and not just this session's share of it.
 */
export async function runProtocol(
  system: OpenMMSystem,
  outputs: RunOutputs,
  settings: ProtocolSettings,
  hooks: ProtocolHooks,
): Promise<RunRecord> {
  const names = stageNames(settings);
  const stages: StageResult[] = [];
  const startedAt = new Date().toISOString();
  let minimisedEnergy: number | null = null;

  // --- Equilibration ------------------------------------------------------
  const equilibrationNames = artefactNames(names.equilibration);
  const alreadyEquilibrated = await outputs.exists(equilibrationNames.checkpoint);
  if (alreadyEquilibrated) {
    stages.push({
      name: names.equilibration,
      steps: 0,
      frames: 0,
      logRows: 0,
      elapsed: 0,
      nsPerDay: 0,
      skipped: true,
    });
  } else {
    const engine = await hooks.engineFor("equilibration", null);
    // Restraints are already on the system by this point: the minimisation is
    // restrained, which is finding 10 in docs/NOTEBOOK_PARITY.md and the
    // opposite of what the stage order suggests.
    minimisedEnergy = await engine.minimise(
      settings.equilibration.minimisationSteps,
      settings.equilibration.minimisationTolerance ?? 10,
    );
    engine.setVelocities(await hooks.initialVelocities(engine));
    const result = await runStage(
      engine,
      outputs,
      names.equilibration,
      settings.equilibration,
      {
        ...(hooks.onReport === undefined ? {} : { onReport: hooks.onReport }),
        ...(hooks.topologyPdb === undefined ? {} : { topologyPdb: hooks.topologyPdb }),
      },
    );
    stages.push(result);
    hooks.onStage?.(result);
  }

  // --- Production, by strides --------------------------------------------
  let carried: OpenMMState | null = null;
  for (const [index, base] of names.strides.entries()) {
    const strideNames = artefactNames(base);
    if (await outputs.exists(strideNames.checkpoint)) {
      const skipped: StageResult = {
        name: base,
        steps: 0,
        frames: 0,
        logRows: 0,
        elapsed: 0,
        nsPerDay: 0,
        skipped: true,
      };
      stages.push(skipped);
      hooks.onStage?.(skipped);
      // Its checkpoint is what the next stride starts from.
      carried = null;
      continue;
    }
    if (carried === null) {
      // The previous stride's checkpoint, or equilibration's for the first.
      const previous =
        index === 0
          ? equilibrationNames.checkpoint
          : artefactNames(names.strides[index - 1] as string).checkpoint;
      const text = await outputs.read(previous);
      if (text === undefined) {
        throw new Error(
          `${base} needs ${previous} to start from, and it is not there. A ` +
            "production stride continues the previous one's state; running " +
            "them out of order is not supported.",
        );
      }
      const { parseState } = await import("./state.js");
      carried = parseState(text);
    }
    const engine = await hooks.engineFor("production", carried);
    const result = await runStage(engine, outputs, base, settings.production, {
      ...(hooks.onReport === undefined ? {} : { onReport: hooks.onReport }),
      ...(hooks.topologyPdb === undefined ? {} : { topologyPdb: hooks.topologyPdb }),
    });
    stages.push(result);
    hooks.onStage?.(result);
    carried = null;
  }

  const record: RunRecord = {
    jobName: settings.jobName,
    seed: settings.seed,
    settings,
    particleCount: system.particleCount,
    stages,
    minimisedEnergy,
    artefacts: [
      ...Object.values(equilibrationNames),
      ...names.strides.flatMap((base) => Object.values(artefactNames(base))),
      "run.json",
    ],
    startedAt,
    finishedAt: new Date().toISOString(),
  };
  await outputs.write("run.json", `${JSON.stringify(record, null, 1)}\n`);
  return record;
}
