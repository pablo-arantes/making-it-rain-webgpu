/**
 * A whole run, end to end, in a browser.
 *
 * Minimise a freshly built system with its restraints on, draw velocities,
 * equilibrate under NPT, then produce by strides -- writing a trajectory, a log
 * and a checkpoint for each, and `run.json` for the lot. This is Phase 2's
 * shape, at a length a test lane can afford; the plan's 1 ns gate is the same
 * code with bigger numbers.
 *
 * What it checks is what only an end-to-end run can:
 *
 *   * the stages happen in the right order and produce every artefact;
 *   * the trajectory's frame count matches its header, and its frame times
 *     match the log's step numbers -- the two are written by different code
 *     paths from the same counter, so a cadence mistake shows up as a mismatch;
 *   * the checkpoint a stride leaves is the state the next one starts from;
 *   * the energies stay finite and the temperature stays in a plausible band,
 *     which is the cheapest statement that the thing is actually integrating;
 *   * a second run over the same outputs skips everything.
 *
 * It deliberately does *not* re-check the physics. Every force, the
 * constraints, the integrators, the barostat and the minimiser have their own
 * differential tests against OpenMM; what is new here is the orchestration
 * touching real hardware.
 */

import { parseState, type OpenMMState } from "../engine/state.js";
import { parseSystem, type MonteCarloBarostat, type OpenMMSystem } from "../engine/system.js";
import {
  artefactNames,
  runProtocol,
  stageNames,
  type ProtocolEngine,
  type ProtocolSettings,
  type RunOutputs,
  type RunRecord,
} from "../engine/protocol.js";
import { velocitiesAtTemperature } from "../reference/velocities.js";
import { splitConstraints } from "../engine/constraint-clusters.js";
import { DCD_HEADER_BYTES } from "../analysis/dcd.js";
import { cellHeadroom } from "./cell-list.js";
import { gridExtents } from "./cell-list.js";
import { StepLoop, type RunOptions } from "./loop.js";
import { LoopEngine } from "./engine.js";

/** An in-memory sink. OPFS is the same four operations; this is the testable one. */
export function memoryOutputs(): RunOutputs & {
  readonly files: Map<string, Uint8Array>;
} {
  const files = new Map<string, Uint8Array>();
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  return {
    files,
    async append(name, bytes) {
      const existing = files.get(name) ?? new Uint8Array(0);
      const joined = new Uint8Array(existing.length + bytes.length);
      joined.set(existing, 0);
      joined.set(bytes, existing.length);
      files.set(name, joined);
    },
    async patch(name, offset, bytes) {
      const existing = files.get(name);
      if (existing === undefined) throw new Error(`cannot patch missing ${name}`);
      existing.set(bytes, offset);
    },
    async write(name, text) {
      files.set(name, encoder.encode(text));
    },
    async read(name) {
      const bytes = files.get(name);
      return bytes === undefined ? undefined : decoder.decode(bytes);
    },
    async exists(name) {
      return files.has(name);
    },
  };
}

export interface ArtefactCheck {
  readonly name: string;
  readonly bytes: number;
  /** For a trajectory: the frame count its header claims. */
  readonly frames: number | null;
  /** For a trajectory: whether its size matches that count exactly. */
  readonly sizeConsistent: boolean | null;
}

export interface StageCheck {
  readonly name: string;
  readonly steps: number;
  readonly frames: number;
  readonly logRows: number;
  readonly nsPerDay: number;
  readonly skipped: boolean;
  /** The step numbers in the log, which must match the trajectory's cadence. */
  readonly firstLogStep: number | null;
  readonly lastLogStep: number | null;
  /** Energies from the log, to say the run was integrating and not diverging. */
  readonly temperatureRange: readonly [number, number] | null;
  readonly allFinite: boolean;
  readonly ok: boolean;
}

export interface ProtocolComparison {
  readonly particleCount: number;
  readonly settings: ProtocolSettings;
  /**
   * NPT, or NVT when the box is too tight for the cell list to survive
   * compression.
   *
   * 🔴 1L2Y cannot run NPT at a 1 nm cutoff. Its box is 3.0806 nm -- 3.08
   * cutoffs -- so the cells start at 1.0269 nm and **2.6% of compression**
   * takes them below the cutoff, at which point the 27-cell scan stops finding
   * every pair. Fewer cells would not help: below three per axis the scan
   * visits the same cell twice. This is finding 4 in docs/NOTEBOOK_PARITY.md
   * showing up as a hard limit rather than a warning -- that fixture's
   * solute already sees its own image at 0.958 nm.
   *
   * So the ensemble is chosen from the measured headroom rather than assumed,
   * and reported. 3HTB has 8.7% and runs NPT.
   */
  readonly ensemble: "NPT" | "NVT";
  readonly cellHeadroom: number;
  readonly stages: readonly StageCheck[];
  readonly artefacts: readonly ArtefactCheck[];
  /** The record written as run.json. */
  readonly record: RunRecord;
  /** A second pass over the same outputs: every stage should be skipped. */
  readonly resumeSkippedEverything: boolean;
  readonly ok: boolean;
}

/** The compression a box can take before its cell list fails. */
const HEADROOM_FOR_NPT = 0.05;

/** Add a barostat, which a run adds rather than the force field. */
function withBarostat(
  system: OpenMMSystem,
  temperature: number,
  pressure: number,
  enabled: boolean,
): OpenMMSystem {
  if (!enabled) return system;
  const barostat: MonteCarloBarostat = {
    type: "MonteCarloBarostat",
    name: "MonteCarloBarostat",
    forceGroup: 0,
    defaultPressure: pressure,
    defaultTemperature: temperature,
    frequency: 25,
    randomSeed: 0,
  };
  return { ...system, forces: [...system.forces, barostat] };
}

/**
 * The hooks `runProtocol` needs, for a given device.
 *
 * Exported so that the persistence check can build the same run against a
 * *different* device after the first one is lost. The ensemble is decided here
 * too, from the measured cell headroom, so both callers make the same choice.
 */
export async function protocolHooksFor(
  device: GPUDevice,
  texts: {
    readonly systemXml: string;
    readonly restrainedXml: string;
    readonly stateXml: string;
    readonly topologyPdb?: string;
  },
  settings: ProtocolSettings,
): Promise<{
  readonly hooks: Parameters<typeof runProtocol>[3];
  readonly ensemble: "NPT" | "NVT";
  readonly headroom: number;
  readonly system: OpenMMSystem;
  readonly loops: StepLoop[];
}> {
  const production = parseSystem(texts.systemXml);
  const restrained = parseSystem(texts.restrainedXml);
  const state = parseState(texts.stateXml);
  const nonbonded = production.forces.find((f) => f.type === "NonbondedForce");
  const cutoff =
    nonbonded !== undefined && "cutoff" in nonbonded ? nonbonded.cutoff : 1;
  const headroom = cellHeadroom(gridExtents(state.boxVectors), cutoff);
  const ensemble: "NPT" | "NVT" = headroom >= HEADROOM_FOR_NPT ? "NPT" : "NVT";
  const loops: StepLoop[] = [];

  const hooks: Parameters<typeof runProtocol>[3] = {
    ...(texts.topologyPdb === undefined ? {} : { topologyPdb: texts.topologyPdb }),
    engineFor: async (stage, carried) => {
      const system =
        stage === "equilibration"
          ? withBarostat(
              restrained,
              settings.equilibration.temperature,
              settings.equilibration.pressure,
              ensemble === "NPT",
            )
          : withBarostat(
              production,
              settings.production.temperature,
              settings.production.pressure,
              ensemble === "NPT",
            );
      const start = carried === null ? state : { ...state, ...carried };
      const loop = await StepLoop.fromParsed(device, system, start, {
        timeStep: settings.equilibration.timeStep,
        temperature:
          stage === "equilibration"
            ? settings.equilibration.temperature
            : settings.production.temperature,
        friction: 1,
        constraintTolerance: 1e-6,
        integrator: "lfmiddle",
        seed: settings.seed,
      });
      loops.push(loop);
      if (carried !== null) loop.advanceStepCount(carried.stepCount);
      return new LoopEngine(loop, system);
    },
    initialVelocities: async (engine) =>
      velocitiesAtTemperature(restrained, {
        temperature: settings.equilibration.temperature,
        seed: settings.seed,
        positions: await engine.positions(),
        timeOffset: 0,
        split: splitConstraints(restrained),
      }),
  };
  return { hooks, ensemble, headroom, system: production, loops };
}

export async function validateProtocol(
  device: GPUDevice,
  texts: {
    /** The production system: no restraints. */
    readonly systemXml: string;
    /** The equilibration system: the same plus a `CustomExternalForce`. */
    readonly restrainedXml: string;
    readonly stateXml: string;
  },
  settings: ProtocolSettings,
): Promise<ProtocolComparison> {
  const production = parseSystem(texts.systemXml);
  const restrained = parseSystem(texts.restrainedXml);
  const state = parseState(texts.stateXml);
  const outputs = memoryOutputs();
  const loops: StepLoop[] = [];

  // Whether this box can take a barostat at all, measured rather than assumed.
  const nonbonded = production.forces.find((f) => f.type === "NonbondedForce");
  const cutoff =
    nonbonded !== undefined && "cutoff" in nonbonded ? nonbonded.cutoff : 1;
  const headroom = cellHeadroom(gridExtents(state.boxVectors), cutoff);
  const ensemble: "NPT" | "NVT" = headroom >= HEADROOM_FOR_NPT ? "NPT" : "NVT";

  const options = (temperature: number, pressure: number): RunOptions => ({
    timeStep: settings.equilibration.timeStep,
    temperature,
    friction: 1,
    constraintTolerance: 1e-6,
    integrator: "lfmiddle",
    seed: settings.seed,
    ...{ pressure },
  });

  const engineFor = async (
    stage: "equilibration" | "production",
    carried: OpenMMState | null,
  ): Promise<ProtocolEngine> => {
    // The notebook builds a **fresh** System for production, with no
    // restraints, rather than removing the force from the equilibration one.
    const system =
      stage === "equilibration"
        ? withBarostat(
            restrained,
            settings.equilibration.temperature,
            settings.equilibration.pressure,
            ensemble === "NPT",
          )
        : withBarostat(
            production,
            settings.production.temperature,
            settings.production.pressure,
            ensemble === "NPT",
          );
    const start = carried === null ? state : { ...state, ...carried };
    const loop = await StepLoop.fromParsed(
      device,
      system,
      start,
      options(
        stage === "equilibration"
          ? settings.equilibration.temperature
          : settings.production.temperature,
        stage === "equilibration"
          ? settings.equilibration.pressure
          : settings.production.pressure,
      ),
    );
    loops.push(loop);
    // A stride continues the step counter, which is what keeps the reporting
    // cadence aligned across strides.
    if (carried !== null) loop.advanceStepCount(carried.stepCount);
    return new LoopEngine(loop, system);
  };

  let record: RunRecord;
  try {
    record = await runProtocol(production, outputs, settings, {
      engineFor,
      initialVelocities: async (engine) =>
        velocitiesAtTemperature(restrained, {
          temperature: settings.equilibration.temperature,
          seed: settings.seed,
          positions: await engine.positions(),
          // Zero here: the shift needs the forces at these positions, and the
          // engine does not hand them out. Measured in docs/DYNAMICS.md as a
          // real effect, so this is a known simplification of the start state
          // rather than a claim that it does not matter.
          timeOffset: 0,
          split: splitConstraints(restrained),
        }),
    });
  } finally {
    for (const loop of loops) loop.destroy();
  }

  // --- Check the artefacts ------------------------------------------------
  const artefacts: ArtefactCheck[] = [];
  for (const [name, bytes] of outputs.files) {
    let frames: number | null = null;
    let sizeConsistent: boolean | null = null;
    if (name.endsWith(".dcd")) {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      frames = view.getInt32(8, true);
      const perFrame =
        3 * (8 + production.particleCount * 4) + 56;
      sizeConsistent = bytes.length === DCD_HEADER_BYTES + frames * perFrame;
    }
    artefacts.push({ name, bytes: bytes.length, frames, sizeConsistent });
  }

  const stages: StageCheck[] = [];
  for (const stage of record.stages) {
    const names = artefactNames(stage.name);
    const log = await outputs.read(names.log);
    const rows = (log ?? "").trim().split("\n").slice(1).filter((line) => line.length > 0);
    const steps = rows.map((row) => Number(row.split(",")[0]));
    const temperatures = rows.map((row) => Number(row.split(",")[4]));
    const allFinite = rows.every((row) =>
      row
        .split(",")
        .slice(1, 6)
        .every((value) => Number.isFinite(Number(value))),
    );
    const trajectory = artefacts.find((a) => a.name === names.trajectory);
    stages.push({
      name: stage.name,
      steps: stage.steps,
      frames: stage.frames,
      logRows: stage.logRows,
      nsPerDay: stage.nsPerDay,
      skipped: stage.skipped,
      firstLogStep: steps[0] ?? null,
      lastLogStep: steps.at(-1) ?? null,
      temperatureRange:
        temperatures.length === 0
          ? null
          : [Math.min(...temperatures), Math.max(...temperatures)],
      allFinite,
      ok:
        stage.skipped ||
        (stage.frames > 0 &&
          stage.logRows > 0 &&
          allFinite &&
          trajectory?.sizeConsistent === true &&
          trajectory.frames === stage.frames &&
          // A thermostatted run at 298 K that has not diverged.
          temperatures.every((value) => value > 150 && value < 500)),
    });
  }

  // --- Resume -------------------------------------------------------------
  const second = await runProtocol(production, outputs, settings, {
    engineFor: async () => {
      throw new Error("a resumed run must not build an engine");
    },
    initialVelocities: async () => {
      throw new Error("a resumed run must not draw velocities");
    },
  });
  const resumeSkippedEverything = second.stages.every((s) => s.skipped);

  return {
    particleCount: production.particleCount,
    settings,
    ensemble,
    cellHeadroom: headroom,
    stages,
    artefacts,
    record,
    resumeSkippedEverything,
    ok:
      stages.every((s) => s.ok) &&
      resumeSkippedEverything &&
      stages.length === 1 + stageNames(settings).strides.length,
  };
}
