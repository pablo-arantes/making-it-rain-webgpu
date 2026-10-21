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

  const perStepGpu = profile.totalNanoseconds / repeats / 1000;
  const perStepWall = profile.wallNanoseconds / repeats / 1000;
  return {
    available: true,
    particleCount: system.particleCount,
    repeats,
    perStepGpuMicroseconds: perStepGpu,
    perStepWallMicroseconds: perStepWall,
    // What the wall-clock contains that the kernels do not.
    overheadShare:
      perStepWall > 0 ? Math.max(0, (perStepWall - perStepGpu) / perStepWall) : 0,
    kernels: profileShares(profile).map((entry) => ({
      label: entry.label,
      share: entry.share,
      microseconds: entry.nanoseconds / repeats / 1000,
    })),
    impliedStepsPerSecond: perStepWall > 0 ? 1e6 / perStepWall : 0,
  };
}
