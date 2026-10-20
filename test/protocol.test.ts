/**
 * The protocol's orchestration, against a stub engine.
 *
 * No dynamics here on purpose. What this tests is the part that has nothing to
 * do with physics and everything to do with getting a run right: when reports
 * land, whether the step counter continues across strides, which stages a
 * resumed run skips, and what ends up in `run.json`. Those are exactly the
 * things a GPU test cannot isolate, and a stub makes them deterministic.
 */

import { describe, expect, it } from "vitest";
import {
  artefactNames,
  runProtocol,
  runStage,
  stageNames,
  stepsForInterval,
  stepsForStage,
  type ProtocolEngine,
  type ProtocolSettings,
  type RunOutputs,
} from "../src/engine/protocol.js";
import { parseState } from "../src/engine/state.js";
import type { OpenMMSystem, Vec3 } from "../src/engine/system.js";

/** An in-memory sink, which is all `RunOutputs` asks for. */
function memoryOutputs(): RunOutputs & { readonly files: Map<string, Uint8Array> } {
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

/** An engine that counts steps and reports plausible numbers. */
function stubEngine(particleCount: number, startStep = 0): ProtocolEngine & {
  readonly calls: string[];
  stepsRun: number;
} {
  const calls: string[] = [];
  let steps = startStep;
  const positions = Float64Array.from(
    { length: particleCount * 3 },
    (_unused, i) => (i % 7) * 0.3,
  );
  const velocities = new Float64Array(particleCount * 3);
  let box: Vec3 = [3, 3, 3] as unknown as Vec3;
  return {
    calls,
    get stepsRun() {
      return steps;
    },
    set stepsRun(value: number) {
      steps = value;
    },
    particleCount,
    get steps() {
      return steps;
    },
    async positions() {
      return positions;
    },
    async velocities() {
      return velocities;
    },
    boxLengths() {
      return box;
    },
    setPositions() {
      calls.push("setPositions");
    },
    setVelocities() {
      calls.push("setVelocities");
    },
    setBox(value) {
      box = value;
    },
    async run(count) {
      calls.push(`run:${count}`);
      steps += count;
    },
    async sample() {
      return {
        potentialEnergy: -1000 - steps,
        kineticEnergy: 500,
        totalEnergy: -500 - steps,
        temperature: 298,
      };
    },
    async minimise(maxIterations) {
      calls.push(`minimise:${maxIterations}`);
      return -12345;
    },
  };
}

const stage = {
  nanoseconds: 0.02, // 10,000 steps at 2 fs
  timeStep: 0.002,
  temperature: 298,
  pressure: 1,
  trajectoryInterval: 4, // 2,000 steps
  logInterval: 2, // 1,000 steps
};

describe("interval arithmetic", () => {
  it("converts picoseconds to steps", () => {
    expect(stepsForInterval(10, 0.002, "x")).toBe(5000);
    expect(stepsForStage(1, 0.002)).toBe(500_000);
  });

  it("refuses an interval the timestep does not divide", () => {
    // 🔴 10 ps at 3 fs is 3333.33 steps. Rounding it makes every frame time
    // wrong by a growing amount, and the notebook offers 3 fs.
    expect(() => stepsForInterval(10, 0.003, "a trajectory interval")).toThrow(
      /not a whole number/,
    );
  });

  it("refuses an interval shorter than a step", () => {
    expect(() => stepsForInterval(0.001, 0.002, "x")).toThrow(/whole number/);
  });
});

describe("runStage", () => {
  it("reports on multiples of the interval and not at step zero", async () => {
    const engine = stubEngine(4);
    const outputs = memoryOutputs();
    const result = await runStage(engine, outputs, "job_equil", stage);
    // 10,000 steps: frames at 2000..10000 is five, rows at 1000..10000 is ten.
    expect(result.steps).toBe(10_000);
    expect(result.frames).toBe(5);
    expect(result.logRows).toBe(10);
    // Nothing at step zero, which is OpenMM's behaviour and a frame count will
    // show: `describeNextReport` returns `interval - currentStep % interval`.
    const log = (await outputs.read("job_equil.log")) as string;
    const rows = log.trim().split("\n").slice(1);
    expect(rows[0]!.startsWith("1000,")).toBe(true);
    expect(rows.at(-1)!.startsWith("10000,")).toBe(true);
  });

  it("keeps the cadence on the absolute step counter", async () => {
    // A stride that starts at 10,000 must report at 12,000, not at 2,000 into
    // itself -- which is the same thing here but would not be if the stage
    // length were not a multiple of the interval.
    const engine = stubEngine(4, 10_000);
    const outputs = memoryOutputs();
    await runStage(engine, outputs, "job_prod_2", stage);
    const log = (await outputs.read("job_prod_2.log")) as string;
    const rows = log.trim().split("\n").slice(1);
    expect(rows[0]!.startsWith("11000,")).toBe(true);
    expect(rows.at(-1)!.startsWith("20000,")).toBe(true);
  });

  it("writes a trajectory whose header knows how many frames it has", async () => {
    const engine = stubEngine(4);
    const outputs = memoryOutputs();
    const result = await runStage(engine, outputs, "job_equil", stage);
    const dcd = outputs.files.get("job_equil.dcd") as Uint8Array;
    const view = new DataView(dcd.buffer, dcd.byteOffset, dcd.byteLength);
    expect(view.getInt32(8, true)).toBe(result.frames);
    // 🔴 ISTART is the step of the *first frame*, not where the stage began.
    // OpenMM's DCDReporter builds its file inside the first `report()` call, by
    // which time `currentStep` is already 2,000 -- and passing the stage's
    // start instead leaves both this and NSTEP one interval low, which a reader
    // turns into a time axis shifted by a frame.
    expect(view.getInt32(12, true)).toBe(2_000);
    expect(view.getInt32(16, true)).toBe(2_000); // NSAVC, the interval
    // And the last step recorded, which is what OpenMM patches at offset 20.
    expect(view.getInt32(20, true)).toBe(10_000);
  });

  it("writes a checkpoint OpenMM's own format can carry", async () => {
    const engine = stubEngine(4);
    const outputs = memoryOutputs();
    await runStage(engine, outputs, "job_equil", stage);
    const checkpoint = parseState((await outputs.read("job_equil.rst")) as string);
    expect(checkpoint.stepCount).toBe(10_000);
    expect(checkpoint.time).toBeCloseTo(20, 9);
    expect(checkpoint.particleCount).toBe(4);
    expect(checkpoint.velocities).toBeDefined();
  });

  it("skips a stage whose checkpoint exists, when asked", async () => {
    const outputs = memoryOutputs();
    await outputs.write("job_prod_1.rst", "<State/>");
    const engine = stubEngine(4);
    const result = await runStage(engine, outputs, "job_prod_1", stage, {
      resume: true,
    });
    expect(result.skipped).toBe(true);
    expect(engine.calls).toEqual([]);
  });
});

describe("runProtocol", () => {
  const system = { particleCount: 4 } as unknown as OpenMMSystem;
  const settings: ProtocolSettings = {
    jobName: "job",
    seed: 0x5eed,
    equilibration: {
      ...stage,
      minimisationSteps: 10_000,
      restraintForceConstant: 700,
    },
    production: { ...stage, strides: 2 },
  };

  const hooks = (engines: ProtocolEngine[]) => ({
    engineFor: async (
      which: "equilibration" | "production",
    ): Promise<ProtocolEngine> => {
      const engine = stubEngine(4, which === "equilibration" ? 0 : 10_000);
      engines.push(engine);
      return engine;
    },
    initialVelocities: async (engine: ProtocolEngine): Promise<Float64Array> =>
      new Float64Array(engine.particleCount * 3),
  });

  it("minimises before drawing velocities, and before any dynamics", async () => {
    const engines: ProtocolEngine[] = [];
    const outputs = memoryOutputs();
    await runProtocol(system, outputs, settings, hooks(engines));
    const calls = (engines[0] as ReturnType<typeof stubEngine>).calls;
    // 🔴 The order is the point. Restraints are already on the system, so the
    // minimisation is restrained -- finding 10 -- and the velocities are drawn
    // from the minimised structure, not from the uploaded one.
    expect(calls[0]).toBe("minimise:10000");
    expect(calls[1]).toBe("setVelocities");
    expect(calls[2]!.startsWith("run:")).toBe(true);
  });

  it("runs every stride and names them as the notebook does", async () => {
    const outputs = memoryOutputs();
    const record = await runProtocol(system, outputs, settings, hooks([]));
    expect(stageNames(settings).strides).toEqual(["job_prod_1", "job_prod_2"]);
    expect(record.stages.map((s) => s.name)).toEqual([
      "job_equil",
      "job_prod_1",
      "job_prod_2",
    ]);
    for (const base of ["job_equil", "job_prod_1", "job_prod_2"]) {
      for (const name of Object.values(artefactNames(base))) {
        expect(outputs.files.has(name), name).toBe(true);
      }
    }
  });

  it("records the run, including what it skipped", async () => {
    const outputs = memoryOutputs();
    await runProtocol(system, outputs, settings, hooks([]));
    const record = JSON.parse((await outputs.read("run.json")) as string) as {
      seed: number;
      minimisedEnergy: number;
      stages: { name: string; skipped: boolean }[];
      settings: ProtocolSettings;
    };
    expect(record.seed).toBe(0x5eed);
    expect(record.minimisedEnergy).toBe(-12345);
    expect(record.settings.equilibration.restraintForceConstant).toBe(700);

    // A second run over the same outputs skips everything, and says so rather
    // than omitting the stages -- the record describes the trajectory on disk,
    // not this session's share of it.
    const again = await runProtocol(system, outputs, settings, hooks([]));
    expect(again.stages.every((s) => s.skipped)).toBe(true);
    expect(again.stages).toHaveLength(3);
    expect(again.minimisedEnergy).toBeNull();
  });

  it("starts a stride from the previous one's checkpoint", async () => {
    const outputs = memoryOutputs();
    const seen: (string | null)[] = [];
    await runProtocol(system, outputs, settings, {
      engineFor: async (which, state) => {
        seen.push(state === null ? null : `state:${state.stepCount}`);
        return stubEngine(4, which === "equilibration" ? 0 : 10_000);
      },
      initialVelocities: async () => new Float64Array(12),
    });
    // Equilibration starts from nothing; the first stride from equilibration's
    // checkpoint at 10,000 steps; the second from the first stride's at 20,000.
    expect(seen).toEqual([null, "state:10000", "state:20000"]);
  });

  it("refuses a stride whose predecessor left no checkpoint", async () => {
    const outputs = memoryOutputs();
    // Equilibration "done" but its checkpoint missing: the only way to reach
    // this is by deleting a file mid-run, and it should say so.
    await outputs.write("job_equil.rst", "<State/>");
    const bad = {
      ...outputs,
      read: async (name: string) =>
        name === "job_equil.rst" ? undefined : outputs.read(name),
    };
    await expect(
      runProtocol(system, bad, settings, hooks([])),
    ).rejects.toThrow(/needs job_equil.rst/);
  });
});
