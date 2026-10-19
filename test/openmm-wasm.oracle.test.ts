import { describe, expect, it } from "vitest";
import { access } from "node:fs/promises";

/**
 * The oracle lane: real OpenMM, compiled to WebAssembly, double-precision
 * Reference platform, as the acceptance reference for every force kernel.
 *
 * It is not wired up yet -- Phase 1 builds it, from the recipe in
 * docs/PRIOR_ART.md section 1 -- and this test exists now so the lane is a
 * visible gap rather than an absent one.  `AGENTS.md` is explicit that the
 * WebAssembly OpenMM is a test fixture and a verification tool, never a
 * fallback and never something that runs dynamics for a user.
 */
const MODULE_PATH = "vendor/openmm-wasm/openmm.wasm";

const present = await access(MODULE_PATH).then(
  () => true,
  () => false,
);

// stderr, not `console.warn`, which Vitest intercepts and does not print.
if (!present) {
  process.stderr.write(
    `\n  Oracle lane NOT WIRED UP: ${MODULE_PATH} is absent.\n` +
      "  Phase 1 builds it; docs/PRIOR_ART.md section 1 has the recipe. Until then\n" +
      "  the reference numbers come from tools/build_reference_system.py.\n\n",
  );
}

describe("the OpenMM WebAssembly oracle", () => {
  it.skipIf(!present)("is available to compare against", () => {
    expect(present).toBe(true);
  });

  it(
    present ? "is present" : "is absent until Phase 1 builds it, and said so",
    () => {
      expect(typeof present).toBe("boolean");
    },
  );
});
