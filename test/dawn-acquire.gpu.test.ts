/**
 * Which stage of getting a Dawn device segfaults **under Vitest**.
 *
 * 🔴 A bisect, not a check. Standalone on `ubuntu-latest` the identical
 * sequence completes and all seven precision checks pass; under Vitest the
 * worker dies — `Worker exited unexpectedly` with the fork pool, and a bare
 * `Segmentation fault` (exit 139) with threads, which is the same fault with
 * the signal no longer hidden behind a dead child. Nothing reports before it,
 * so it happens while this file is being set up rather than inside a test.
 *
 * `acquireDawn` is shared with `tools/dawn-probe.ts` so the two runners execute
 * one copy of the sequence. The lines go out with `writeSync(1, ...)`: Vitest
 * buffers `console.log` per test, and a buffered line is lost when the process
 * takes a signal — so the last line printed is the stage that killed it.
 */

import { writeSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { acquireDawn } from "../tools/dawn-stages.js";
// 🔴 Static, and that is the experiment. The dynamic form of this import --
// after a Dawn device existed -- segfaulted on the runner after 2.7 seconds,
// before the first check started. Static means Vitest transforms the whole
// kernel graph *before* `webgpu` is loaded, which is the order
// `tools/dawn-probe.ts` uses and the order that survives standalone.
import * as checks from "../src/gpu/checks.js";

const say = (message: string): void => {
  writeSync(1, `STAGE ${message}\n`);
};

describe.skipIf(process.env["MIR_GPU_TESTS"] !== "1")("acquiring a Dawn device", () => {
  it("names the stage it reaches", async () => {
    say("test started");
    // 🔴 A *catchable* failure is not what this file hunts, and must not fail
    // the lane. On macOS the `webgpu` import throws -- no published binary
    // loads there -- and `npm run test:gpu` exits 0 today precisely because
    // `test/dawn.ts` catches that. Turning it into a failure would break the
    // lane on the only machine that can develop it, to chase a crash that only
    // happens on Linux. The crash itself is a signal and cannot be caught, so
    // nothing here hides it.
    let device: GPUDevice | null = null;
    try {
      device = await acquireDawn(say);
    } catch (error) {
      say(`unavailable (catchable): ${String(error).split("\n")[0]}`);
      expect(true).toBe(true);
      return;
    }
    say(`acquire returned ${device === null ? "null" : "a device"}`);
    // Deliberately weak: this file exists to survive, not to assert. If the
    // sequence completes at all, the fault is later than this.
    expect(device === null || typeof device === "object").toBe(true);

    // 🔴 Acquisition passed in both pools, so the fault is in the checks as
    // *Vitest* runs them -- the same checks pass standalone on the same
    // runner. `MIR_DAWN_BISECT=1` runs them here, one at a time, naming each
    // before it starts, so the last line printed is the one that took the
    // process down. Off by default: this is a diagnostic and the lane already
    // runs these checks properly in `precision.gpu.test.ts`.
    if (device !== null && process.env["MIR_DAWN_BISECT"] === "1") {
      say("checks module already imported statically");
      const ordered = [
        ["fixed-point exactness", checks.checkFixedPointExactness],
        ["fixed-point carry", checks.checkFixedPointCarry],
        ["position stagnation", checks.checkPositionStagnation],
        ["position random walk", checks.checkPositionRandomWalk],
        ["Dekker diagnostic", checks.checkDekkerDiagnostic],
        ["trig diagnostic", checks.checkTrigDiagnostic],
        ["mixed-radix FFT", checks.checkFft],
      ] as const;
      for (const [name, run] of ordered) {
        say(`--> ${name}`);
        const result = await run(device);
        say(`<-- ${name}: ok=${String(result.ok)}`);
      }
      say("every check survived under vitest");
    }
    say("test finished");
  });
});
