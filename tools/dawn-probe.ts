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
  // 🔴 Is the crash scale or pattern? `checkFixedPointExactness` pushes 65,536
  // values through 1,024 workgroups into 16 atomic slots, and it is the first
  // check and the one that dies — on llvmpipe, on macOS 15 Metal and on macOS
  // 26 Metal, always at the same call. If a 64-value run survives and a 65,536
  // one does not, it is scale: a dispatch size or a readback Dawn-in-Node does
  // not like. If even 64 dies, it is the call pattern itself, and the size is a
  // red herring. One run answers that; reading `readBuffer` cannot, because it
  // is textbook correct.
  if (process.env["MIR_DAWN_MICRO"] === "1") {
    const { accumulateFixedPoint } = await import("../src/gpu/precision.js");
    for (const count of [64, 1024, 16384, 65536]) {
      for (const slots of [1, 16]) {
        say(`micro --> ${count} values into ${slots} slot(s)`);
        const values = new Float32Array(count).fill(0.25);
        const sums = await accumulateFixedPoint(device, values, slots);
        say(`micro <-- ${count}/${slots}: ${sums.length} slot(s) back`);
      }
    }
    say("micro: every size survived");
  }

  const failures: string[] = [];
  for (const [name, run] of checks) {
    say(`--> ${name}`);
    try {
      const result = await run(device);
      say(`<-- ${name}: ok=${String(result.ok)}`);
      if (!result.ok) failures.push(name);
    } catch (error) {
      // A *catchable* failure is a different animal from the crash, and saying
      // so keeps the two from being confused in the log.
      say(`<!! ${name} threw (not a crash): ${String(error).split("\n")[0]}`);
      failures.push(name);
    }
  }
  say("all checks completed");
  if (failures.length > 0) {
    // 🔴 A gate, not a report. CI runs this *instead of* the Vitest lane, which
    // segfaults next to llvmpipe, so a differential failure on a non-Apple
    // implementation has to fail the build here or it is not checked anywhere.
    say(`FAILED CHECKS: ${failures.join(", ")}`);
    process.exitCode = 1;
  }
}

main().then(
  () => say("exited normally"),
  (error) => {
    say(`FAILED (catchable): ${String(error)}`);
    process.exitCode = 1;
  },
);
