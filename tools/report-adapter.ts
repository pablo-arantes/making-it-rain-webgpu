/**
 * Print this machine's adapter report and precision-check results as JSON.
 *
 *   npm run report:adapter
 *   npm run report:adapter -- docs/measurements/device-<name>.json
 *
 * The committed copies under docs/measurements/ are what the engineering
 * invariants in AGENTS.md cite, so a claim about a device can be traced to the
 * run that produced it.
 */

import { writeFileSync } from "node:fs";
import type { DeviceCheckReport } from "../src/gpu/checks.js";
import { inChrome, unwrap } from "./chrome.js";

const out = process.argv[2];
const { value, consoleErrors } = await inChrome<DeviceCheckReport | { error: string }>(
  "/",
  "__mirReport",
);
const report = unwrap(value);
const payload = { ...report, consoleErrors, capturedAt: new Date().toISOString() };
const text = JSON.stringify(payload, null, 1);
if (out !== undefined) {
  writeFileSync(out, `${text}\n`);
  console.log(`wrote ${out}`);
} else {
  console.log(text);
}
if (!report.ok) {
  console.error("\nAt least one precision check failed on this device.");
  process.exitCode = 1;
}
