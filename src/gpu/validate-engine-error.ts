/**
 * The engine's force error at the coordinates the acceptance thresholds are
 * defined at.
 *
 * 🔴 This exists because the number is cited in `AGENTS.md` and was, for a
 * while, produced by an ad-hoc probe rather than by anything committed. The
 * project's own rule is that a number in `docs/` traces to the run that made
 * it, and a headline figure measured once by hand drifts the moment a kernel
 * changes — which it then did, through four kernel changes, without anyone
 * being able to tell.
 *
 * The coordinates matter as much as the comparison. A fixture's own `state.xml`
 * is not equilibrated, and at raw coordinates a relative force error reports on
 * the input structure rather than on the platform — so the figure is taken at
 * the *same saved frame* `tools/calibrate_thresholds.py` measured OpenMM's
 * platforms at. Equilibration is stochastic; re-running it gives an
 * incomparable structure, which is why the frame is a committed fixture and not
 * a protocol.
 *
 * The comparison itself is `verifySystem`, which is the same code path as the
 * "verify this system" button a user presses. Nothing here is a bespoke
 * harness.
 */

import { verifySystem, type VerifyReport } from "./verify.js";

export interface EngineErrorReport {
  /** Which saved frame this was measured at. */
  readonly state: string;
  readonly particleCount: number;
  /** Relative RMS force difference against our own f64 reference. */
  readonly forceRelativeRms: number;
  readonly energyRelativeError: number;
  readonly deviceEnergy: number;
  readonly referenceEnergy: number;
  readonly absoluteEnergyError: number;
  readonly maxReferenceForce: number;
  readonly ok: boolean;
  /** The whole verify report, so nothing is lost on the way to a file. */
  readonly verify: VerifyReport;
}

export async function validateEngineError(
  device: GPUDevice,
  texts: {
    readonly systemXml: string;
    readonly equilibratedStateXml: string;
    readonly state: string;
  },
): Promise<EngineErrorReport> {
  const verify = await verifySystem(device, {
    systemXml: texts.systemXml,
    stateXml: texts.equilibratedStateXml,
  });
  return {
    state: texts.state,
    particleCount: verify.particleCount,
    forceRelativeRms: verify.forceRelativeRms,
    energyRelativeError: verify.energyRelativeError,
    deviceEnergy: verify.deviceEnergy,
    referenceEnergy: verify.referenceEnergy,
    absoluteEnergyError: Math.abs(verify.deviceEnergy - verify.referenceEnergy),
    maxReferenceForce: verify.maxReferenceForce,
    ok: verify.ok,
    verify,
  };
}
