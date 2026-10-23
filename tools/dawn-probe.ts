/**
 * Where exactly Dawn dies, stage by stage.
 *
 * 🔴 CI's `gpu` job gets an adapter on `ubuntu-latest` — lavapipe's own
 * "maxDynamicUniformBuffersPerPipelineLayout artificially reduced" warnings are
 * in the log — and then the vitest worker exits with
 * `Worker exited unexpectedly` and no assertion reached. A native crash is
 * invisible to `skipIf`, and vitest's fork adds a layer between the fault and
 * the report, so this reproduces the same sequence as `test/dawn.ts` in **one
 * process, with no test runner**, and names each step before taking it.
 *
 * 🔴 Every line goes out with `writeSync(1, ...)`. `console.log` buffers, and a
 * process that dies in native code never flushes — so the last thing printed
 * would be minutes of work before the actual fault. The whole point here is
 * that the final line is the step that killed it.
 *
 *     node --import tsx tools/dawn-probe.ts
 *     gdb -batch -ex run -ex 'thread apply all bt' \
 *         --args node --import tsx tools/dawn-probe.ts
 *
 * Run it under `gdb` for the native stack; run it plain to find the stage.
 */

import { writeSync } from "node:fs";
import { acquireDawn } from "./dawn-stages.js";
import {
  checkDekkerDiagnostic,
  checkFft,
  checkFixedPointCarry,
  checkFixedPointExactness,
  checkPositionRandomWalk,
  checkPositionStagnation,
  checkTrigDiagnostic,
  type CheckResult,
} from "../src/gpu/checks.js";

const say = (message: string): void => {
  writeSync(1, `PROBE ${message}\n`);
};

async function main(): Promise<void> {
  say("start");
  const device = await acquireDawn(say);
  if (device === null) {
    say("no adapter; nothing further to probe");
    return;
  }

  const checks: ReadonlyArray<readonly [string, (d: GPUDevice) => Promise<CheckResult>]> = [
    ["fixed-point exactness", checkFixedPointExactness],
    ["fixed-point carry", checkFixedPointCarry],
    ["position stagnation", checkPositionStagnation],
    ["position random walk", checkPositionRandomWalk],
    ["Dekker diagnostic", checkDekkerDiagnostic],
    ["trig diagnostic", checkTrigDiagnostic],
    ["mixed-radix FFT", checkFft],
  ];
  for (const [name, run] of checks) {
    say(`--> ${name}`);
    try {
      const result = await run(device);
      say(`<-- ${name}: ok=${String(result.ok)}`);
    } catch (error) {
      // A *catchable* failure is a different animal from the crash, and saying
      // so keeps the two from being confused in the log.
      say(`<!! ${name} threw (not a crash): ${String(error).split("\n")[0]}`);
    }
  }
  say("all checks completed");
}

main().then(
  () => say("exited normally"),
  (error) => {
    say(`FAILED (catchable): ${String(error)}`);
    process.exitCode = 1;
  },
);
