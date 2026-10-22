/**
 * Where a step's GPU time goes, per kernel.
 *
 * The engine runs 1L2Y at about 208 steps/s, which is roughly what OpenMM's
 * CPU platform manages on the same system — so there is headroom, and nothing
 * in the code says where. Two guesses were already wrong: readbacks are in the
 * accessors rather than the per-step path, and steps are batched into one
 * submission. This measures instead.
 *
 * 🔴 Reported alongside the wall-clock for the whole submission, because the
 * two answer different questions and the gap between them is itself the
 * finding. Summed GPU time says what the kernels cost; wall-clock says what the
 * caller waited. If wall-clock greatly exceeds the sum, the work is launch
 * overhead and queue bubbles rather than any kernel, and optimising a kernel
 * would move nothing.
 */

import { StepLoop, type RunOptions } from "./loop.js";
import { minimiseOnDevice } from "./minimise.js";
import { parseState } from "../engine/state.js";
import { parseSystem } from "../engine/system.js";
import { canProfile, profileShares, type Profile } from "./profile.js";

export interface ProfileComparison {
  readonly available: boolean;
  readonly reason?: string;
  readonly particleCount: number;
  readonly repeats: number;
  /** Microseconds of summed GPU time for one step. */
  readonly perStepGpuMicroseconds: number;
  /** Microseconds of wall-clock per step for the same submission. */
  readonly perStepWallMicroseconds: number;
  /** The share of GPU time that is not accounted for by any kernel. */
  readonly overheadShare: number;
  readonly kernels: readonly {
    readonly label: string;
    readonly share: number;
    readonly microseconds: number;
  }[];
  /** Implied steps per second from the wall-clock, for comparison with a run. */
  readonly impliedStepsPerSecond: number;
  /**
   * Steps per second over a real run, which is the number that matters.
   *
   * 🔴 The per-kernel breakdown cannot answer this on its own once there is a
   * neighbour list, because `repeats` decides how often the rebuild lands
   * inside it -- and replaying one step's parameter block drives the atoms
   * harder than a run does, so the rebuild fires far more often than it would.
   * Measured over `steadySteps` real steps instead, with the rebuilds counted
   * on the device rather than assumed.
   */
  readonly steadyStepsPerSecond: number;
  /**
   * The same run on the cell-shell path, so the list's effect is measured
   * against the thing it replaced rather than against a number from an earlier
   * build on an unknown machine state.
   */
  readonly shellStepsPerSecond: number;
  /** How far the shell path's atoms moved, for comparison with the list's. */
  readonly shellWorstDisplacement: number;
  /**
   * Potential energy per atom after each run, kJ/mol.
   *
   * 🔴 Here because a throughput number alone cannot tell a fast engine from a
   * diverging one, and a diverging run is *slower*: its atoms spread out, its
   * cells become uneven and its kernels take longer. Two runs from the same
   * state that end at different energies are not two measurements of the same
   * thing.
   */
  readonly listEnergyPerAtom: number;
  readonly shellEnergyPerAtom: number;
  readonly steadySteps: number;
  /** How many neighbour-list rebuilds those steps needed. */
  readonly rebuilds: number;
  /** Steps per rebuild, which is what the skin buys. */
  readonly stepsPerRebuild: number;
  /**
   * The largest displacement since a build the run saw, nm.
   *
   * Against `verletSkin / 2`, which is the threshold: a value just over it
   * means the criterion is doing what it should, and a value far over it means
   * something else moved the atoms.
   */
  readonly worstDisplacement: number;
  /** Whether any atom had more neighbours than the list had room for. */
  readonly overflowed: boolean;
  readonly verletCapacity: number;
  readonly verletSkin: number;
  /** Bytes the neighbour list and its bookkeeping hold. */
  readonly verletByteLength: number;
}

export async function validateProfile(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
  repeats = 40,
): Promise<ProfileComparison> {
  const system = parseSystem(texts.systemXml);
  const empty = {
    particleCount: system.particleCount,
    repeats,
    perStepGpuMicroseconds: 0,
    perStepWallMicroseconds: 0,
    overheadShare: 0,
    kernels: [],
    impliedStepsPerSecond: 0,
    steadyStepsPerSecond: 0,
    shellStepsPerSecond: 0,
    shellWorstDisplacement: 0,
    listEnergyPerAtom: 0,
    shellEnergyPerAtom: 0,
    steadySteps: 0,
    rebuilds: 0,
    stepsPerRebuild: 0,
    worstDisplacement: 0,
    overflowed: false,
    verletCapacity: 0,
    verletSkin: 0,
    verletByteLength: 0,
  };
  if (!canProfile(device)) {
    return {
      ...empty,
      available: false,
      reason:
        "this device does not offer `timestamp-query`, so per-kernel GPU time " +
        "cannot be measured. The feature is optional and the engine does not " +
        "need it; only this measurement does.",
    };
  }

  const state = parseState(texts.stateXml);
  const options: RunOptions = {
    timeStep: 0.002,
    temperature: 298,
    friction: 1,
    seed: 24301,
    integrator: "lfmiddle",
    constraintTolerance: 1e-6,
  };
  // A loop that exists only to be measured: `profileStep` advances the system
  // without advancing its step counter, so this one is discarded afterwards.
  const loop = await StepLoop.fromParsed(device, system, state, options);
  const profile = (await loop.profileStep(repeats)) as Profile;
  // What the query set could actually hold, which may be fewer than asked for.
  const taken = profile.repeats;

  const perStepGpu = profile.totalNanoseconds / taken / 1000;
  const perStepWall = profile.wallNanoseconds / taken / 1000;

  // Then a real run, long enough that the rebuilds in it are representative.
  //
  // 🔴 On a *fresh* loop. `profileStep` advances the system by `repeats` steps
  // all driven from step 0's parameter block -- the same thermostat draw every
  // time -- which is a systematic kick rather than a random one, and it heats
  // the system. On 3HTB that was enough to take the positions to infinity, and
  // the symptom was not a failed check but a neighbour list that rebuilt every
  // step: the displacement test is the first thing that notices a system
  // leaving the box. Measuring throughput on the loop that had just been
  // profiled measured a blown-up system.
  loop.destroy();
  const steadySteps = Math.max(200, Math.min(2000, Math.ceil(4e6 / system.particleCount)));

  const timeRun = async (
    skin: number | undefined,
  ): Promise<{
    seconds: number;
    flags: Awaited<ReturnType<StepLoop["verletFlags"]>>;
    energyPerAtom: number;
  }> => {
    const run = await StepLoop.fromParsed(device, system, state, {
      ...options,
      ...(skin === undefined ? {} : { verletSkin: skin }),
    });
    // 🔴 Minimise first, and this is not tidiness. Timing a run from a fixture
    // state that has not been relaxed times a *diverging* run: 3HTB went from
    // -7.6 kJ/mol an atom to -26,893 over 200 steps on the cell-shell path and
    // -56,929 on the list path, and a diverging run is slower, because its
    // atoms spread out and its cells stop being even. Both numbers were real
    // measurements of nothing. 1L2Y's fixture happens to be relaxed, which is
    // why it looked fine.
    await minimiseOnDevice(run, system, { maxIterations: 200 });
    const began = performance.now();
    await run.run(steadySteps);
    const seconds = (performance.now() - began) / 1000;
    const flags = await run.verletFlags();
    const after = await run.potential(await run.positions());
    run.destroy();
    return { seconds, flags, energyPerAtom: after.energy / system.particleCount };
  };

  const list = await timeRun(undefined);
  // 🔴 The same run on the shell path, from the same starting state. Without
  // this the two numbers come from different builds, and a run that diverges --
  // which the displacement figure reveals, because nothing notices a system
  // leaving the box sooner -- looks like the list's fault rather than the
  // fixture's.
  const shell = await timeRun(0);
  const seconds = list.seconds;
  const flags = list.flags;

  return {
    available: true,
    particleCount: system.particleCount,
    repeats: taken,
    perStepGpuMicroseconds: perStepGpu,
    perStepWallMicroseconds: perStepWall,
    // What the wall-clock contains that the kernels do not.
    overheadShare:
      perStepWall > 0 ? Math.max(0, (perStepWall - perStepGpu) / perStepWall) : 0,
    kernels: profileShares(profile).map((entry) => ({
      label: entry.label,
      share: entry.share,
      microseconds: entry.nanoseconds / taken / 1000,
    })),
    impliedStepsPerSecond: perStepWall > 0 ? 1e6 / perStepWall : 0,
    steadyStepsPerSecond: seconds > 0 ? steadySteps / seconds : 0,
    shellStepsPerSecond: shell.seconds > 0 ? steadySteps / shell.seconds : 0,
    shellWorstDisplacement: shell.flags.worstDisplacement,
    listEnergyPerAtom: list.energyPerAtom,
    shellEnergyPerAtom: shell.energyPerAtom,
    steadySteps,
    rebuilds: flags.rebuilds,
    stepsPerRebuild: flags.rebuilds > 0 ? steadySteps / flags.rebuilds : steadySteps,
    worstDisplacement: flags.worstDisplacement,
    overflowed: flags.overflowed,
    verletCapacity: flags.capacity,
    verletSkin: flags.skin,
    verletByteLength: flags.byteLength,
  };
}
