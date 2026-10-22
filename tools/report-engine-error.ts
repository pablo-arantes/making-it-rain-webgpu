/**
 * Re-measure the engine's force error against the acceptance basis.
 *
 *   npm run report:engine-error
 *   npm run report:engine-error -- docs/measurements/engine-force-error-equilibrated-m3pro.json
 *
 * AGENTS.md's "where the engine sits against it" table cites this, and for a
 * while it was produced by hand — so four kernel changes went past it without
 * anyone being able to tell whether the figure still described the engine that
 * ships. The browser measures; this composes the record, reading OpenMM's own
 * rows from the committed calibration rather than restating them.
 */

import { readFileSync, writeFileSync } from "node:fs";
import type { EngineErrorReport } from "../src/gpu/validate-engine-error.js";
import { inChrome } from "./chrome.js";

const CALIBRATION = "docs/measurements/calibration-L4-opencl-mixed.json";

interface Calibration {
  readonly proposed_thresholds: { readonly force_rel_rms: number };
  readonly platforms: Readonly<
    Record<string, { readonly single_point_vs_reference: { readonly force_rel_rms: number } }>
  >;
}

const out = process.argv[2];
const { value, consoleErrors } = await inChrome<
  EngineErrorReport | { error: string; absent?: boolean }
>("/validate.html?fixture=1L2Y", "__mirEngineError", 1_800_000);

if ("error" in value) {
  console.error(value.error);
  process.exitCode = 1;
} else {
  const calibration = JSON.parse(readFileSync(CALIBRATION, "utf8")) as Calibration;
  const single =
    calibration.platforms["OpenCL/single"]?.single_point_vs_reference.force_rel_rms;
  const double = calibration.platforms["CPU"]?.single_point_vs_reference.force_rel_rms;
  const engine = value.forceRelativeRms;
  const payload = {
    what:
      "The engine's force error at the configuration the acceptance thresholds " +
      "are defined at, so it is comparable with them.",
    date: new Date().toISOString().slice(0, 10),
    produced_by: "npm run report:engine-error, which drives web/validate.html in Chrome",
    host: "Apple M3 Pro, macOS 14, Google Chrome, apple · metal-3",
    state: value.state,
    particleCount: value.particleCount,
    force_relative_rms: {
      _definition:
        "sqrt(sum((a-b)^2) / sum(b^2)) over all 3N components; the identical " +
        "formula in src/gpu/verify.ts and tools/calibrate_thresholds.py, which " +
        "is what makes these comparable",
      webgpu_engine_vs_our_f64_reference: engine,
      openmm_opencl_single_vs_openmm_reference: single,
      openmm_cpu_double_vs_openmm_reference: double,
      engine_better_than_openmm_single_by:
        single === undefined ? null : Number((single / engine).toFixed(2)),
      engine_worse_than_openmm_double_by:
        double === undefined ? null : Number((engine / double).toFixed(2)),
      basis_from: CALIBRATION,
    },
    total_energy: {
      webgpu_engine_relative: value.energyRelativeError,
      webgpu_engine_absolute_kJ_mol: value.absoluteEnergyError,
      device_kJ_mol: value.deviceEnergy,
      our_f64_reference_kJ_mol: value.referenceEnergy,
      note:
        "The asymmetry between forces and total energy is real and is not a " +
        "defect: forces accumulate exactly in 64-bit fixed point, whereas the " +
        "total is a difference of two large cancelling PME terms, so its " +
        "relative error is amplified by the cancellation.",
    },
    mesh: value.verify.mesh,
    alpha: value.verify.alpha,
    cutoff: value.verify.cutoff,
    ok: value.ok,
    consoleErrors,
  };
  const text = JSON.stringify(payload, null, 1);
  if (out === undefined) {
    console.log(text);
  } else {
    writeFileSync(out, `${text}\n`);
    console.log(`wrote ${out}`);
  }
  console.log(
    `force rel. RMS ${engine.toExponential(3)}` +
      (single === undefined ? "" : `, ${(single / engine).toFixed(1)}x inside OpenMM single`),
  );
}
