import { describe, expect, it } from "vitest";
import { dawn } from "./dawn.js";
import {
  checkDekkerDiagnostic,
  checkFixedPointCarry,
  checkFixedPointExactness,
  checkPositionRandomWalk,
  checkPositionStagnation,
  type CheckResult,
} from "../src/gpu/checks.js";

/**
 * The precision primitives on a real device, through Dawn.
 *
 * The same checks run in the browser lane (test/browser/checks.spec.ts). Both
 * call the same functions in `src/gpu/checks.ts` so a result cannot depend on
 * which lane was available.
 */
const status = await dawn();

// Written to stderr at collection time rather than through `console.warn`,
// which Vitest intercepts and does not print in run mode. A GPU lane that is
// silently skipping is indistinguishable from one that is passing, and this
// lane cannot run at all on macOS 14 (see CLAUDE.md), so it has to say so.
if (status.kind !== "ready") {
  process.stderr.write(
    status.kind === "opted-out"
      ? "\n  Dawn lane opted out: set MIR_GPU_TESTS=1 (npm run test:gpu)\n\n"
      : `\n  Dawn lane UNAVAILABLE: ${status.reason}\n` +
        "  The browser lane (npm run test:browser) runs the same checks.\n\n",
  );
}

describe.skipIf(status.kind !== "ready")("precision primitives on Dawn", () => {
  const handles = status.kind === "ready" ? status.handles : null;

  const expectPass = (result: CheckResult): void => {
    // Printing the measurement on failure is the point: a bare boolean would
    // say the kernel is wrong without saying by how much.
    expect(result.ok, `${result.name}\n${JSON.stringify(result.detail, null, 1)}`).toBe(
      true,
    );
  };

  it("accumulates forces exactly under contention", async () => {
    expectPass(await checkFixedPointExactness(handles!.device));
  });

  it("gets the emulated 64-bit carry right", async () => {
    expectPass(await checkFixedPointCarry(handles!.device));
  });

  it("keeps positions moving where f32 stagnates", async () => {
    expectPass(await checkPositionStagnation(handles!.device));
  });

  it("tracks a realistic displacement history", async () => {
    expectPass(await checkPositionRandomWalk(handles!.device));
  });

  it("reports what the backend does with error-free arithmetic", async () => {
    const result = await checkDekkerDiagnostic(handles!.device);
    // A diagnostic: it records rather than gates. Printed so a device that
    // behaves differently from Chrome/Metal is visible in the log.
    console.log(result.name, JSON.stringify(result.detail, null, 1));
    expect(result.ok).toBe(true);
  });
});

describe("the Dawn lane reports its own state", () => {
  it(
    status.kind === "ready"
      ? "has a device"
      : status.kind === "opted-out"
        ? "is opted out (MIR_GPU_TESTS is not 1)"
        : "is unavailable, and said why on stderr",
    () => {
      if (status.kind === "ready") {
        expect(status.handles.device).toBeDefined();
        return;
      }
      // Not a failure: this machine genuinely cannot load Dawn. The reason is
      // on stderr above, and the browser lane covers the same checks.
      expect(status.kind === "opted-out" || status.reason.length > 0).toBe(true);
    },
  );
});
