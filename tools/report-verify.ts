/**
 * Does the GPU engine agree with its own f64 reference on this fixture?
 *
 *   npm run report:verify                          # 1L2Y
 *   npm run report:verify -- 1L2Y-dodecahedron
 *   npm run report:verify -- 3HTB docs/measurements/verify-3HTB.json
 *
 * One question, one command, a few seconds. It drives `web/validate.html` with
 * `?only=verify`, so none of the page's other checks run — without that filter
 * the answer arrives after the force parity, the constraints and the cell list,
 * which on a large fixture is minutes.
 *
 * 🔴 A refusal is a result, not a crash. While the kernels are rectangular-only
 * a triclinic fixture comes back refused, and printing that is how the
 * conversion's progress gets measured: the same command starts returning
 * numbers as the kernels land. It exits non-zero either way so a loop notices.
 *
 * The comparison is `verifySystem`, which is the code path behind the "verify
 * this system" button — not a harness built for this.
 */

import { writeFileSync } from "node:fs";
import type { VerifyReport } from "../src/gpu/verify.js";
import { inChrome } from "./chrome.js";

const fixture = process.argv[2] ?? "1L2Y";
const out = process.argv[3];

const { value, consoleErrors } = await inChrome<
  VerifyReport | { error: string } | { skipped: true }
>(`/validate.html?fixture=${encodeURIComponent(fixture)}&only=verify`, "__mirVerify", 900_000);

if (value === undefined) {
  console.error(`${fixture}: the page never reported a result.`);
  process.exitCode = 1;
} else if ("skipped" in value) {
  console.error(
    `${fixture}: the page skipped the check, which means the \`only=verify\` ` +
      "filter did not match. The name is `verify`.",
  );
  process.exitCode = 1;
} else if ("error" in value) {
  console.log(`${fixture}: REFUSED`);
  console.log(`  ${value.error.split("\n").join("\n  ")}`);
  process.exitCode = 1;
} else {
  const report = value;
  console.log(
    `${fixture}: ${report.particleCount.toLocaleString("en-US")} particles, ` +
      `box ${report.box.map((v) => v.toFixed(3)).join(" x ")} nm, ` +
      `mesh ${report.mesh.join("x")}`,
  );
  console.log(
    `  energy  ${report.energyRelativeError.toExponential(3)} relative ` +
      `(device ${report.deviceEnergy.toFixed(4)}, reference ${report.referenceEnergy.toFixed(4)} kJ/mol)`,
  );
  console.log(`  force   ${report.forceRelativeRms.toExponential(3)} relative RMS`);
  console.log(`  ${report.ok ? "pass" : "FAIL"}`);
  if (consoleErrors.length > 0) {
    console.log(`  console errors: ${consoleErrors.length}`);
  }
  if (out !== undefined) {
    writeFileSync(
      out,
      `${JSON.stringify({ fixture, ...report, consoleErrors, capturedAt: new Date().toISOString() }, null, 1)}\n`,
    );
    console.log(`  wrote ${out}`);
  }
  if (!report.ok) process.exitCode = 1;
}
