/**
 * Benchmark the primitives, on this machine, through a real browser.
 *
 *   npm run bench
 *   npm run bench -- docs/measurements/bench-<name>.json
 *
 * These are engineering measurements from one adapter and one browser build.
 * They are for comparing a warm median on the same machine before and after a
 * change, and they are not performance claims across devices.
 */

import { writeFileSync } from "node:fs";
import type { BenchReport } from "../src/gpu/bench.js";
import { inChrome, unwrap } from "./chrome.js";

const out = process.argv[2];
const { value } = await inChrome<BenchReport | { error: string }>(
  "/?bench=1",
  "__mirBench",
);
const report = unwrap(value);

const named = [report.adapter.vendor, report.adapter.architecture, report.adapter.device]
  .filter(Boolean)
  .join(" ");
console.log(`\n${named || report.adapter.description || "unknown adapter"}\n`);
for (const row of report.rows) {
  console.log(`  ${row.name.padEnd(44)} ${String(row.value).padStart(9)} ${row.unit}`);
}
console.log();

if (out !== undefined) {
  writeFileSync(
    out,
    `${JSON.stringify({ ...report, capturedAt: new Date().toISOString() }, null, 1)}\n`,
  );
  console.log(`wrote ${out}`);
}
