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

  say("importing webgpu");
  const webgpu = (await import("webgpu")) as unknown as {
    create: (flags: string[]) => GPU;
    globals: Record<string, unknown>;
  };
  say("webgpu imported");

  Object.assign(globalThis, webgpu.globals);
  say("globals assigned");

  // The same flag `test/dawn.ts` passes, so the instance is configured the same.
  const gpu = webgpu.create(["enable-dawn-features=allow_unsafe_apis"]);
  say("instance created");

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  say(`requestAdapter returned ${adapter === null ? "null" : "an adapter"}`);
  if (adapter === null) {
    say("no adapter; nothing further to probe");
    return;
  }

  const info = adapter.info as unknown as Record<string, unknown> | undefined;
  say(
    `adapter: vendor=${String(info?.["vendor"])} architecture=${String(
      info?.["architecture"],
    )} device=${String(info?.["device"])}`,
  );
  for (const key of [
    "maxStorageBufferBindingSize",
    "maxBufferSize",
    "maxStorageBuffersPerShaderStage",
    "maxComputeInvocationsPerWorkgroup",
    "maxComputeWorkgroupStorageSize",
  ]) {
    say(`adapter.limits.${key}=${String((adapter.limits as unknown as Record<string, unknown>)[key])}`);
  }

  // 🔴 The step most likely to be it: this asks for raised limits, and a
  // software adapter advertises implausible maxima that Dawn then clamps.
  say("requesting the engine device (raised limits)");
  const { requestEngineDevice } = await import("../src/gpu/device.js");
  const device = await requestEngineDevice(adapter);
  say("device created");

  device.addEventListener?.("uncapturederror", (event) => {
    say(`uncaptured error: ${String((event as unknown as { error?: unknown }).error)}`);
  });
  device.lost
    ?.then((reason) => {
      say(`DEVICE LOST: ${String((reason as unknown as { message?: string }).message)}`);
    })
    .catch(() => {});

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
