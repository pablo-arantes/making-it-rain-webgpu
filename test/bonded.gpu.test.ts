import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { dawn } from "./dawn.js";
import { validateBonded } from "../src/gpu/validate.js";
import { FIXTURE_NAMES, fixture } from "./fixtures.js";

/**
 * The bonded kernels against OpenMM, through Dawn.
 *
 * The same comparison the browser lane runs (test/browser/bonded.spec.ts),
 * calling the same `validateBonded`, so a result cannot depend on which lane
 * was available. On macOS 14 this lane reports itself unavailable and the
 * browser lane is the one that runs; in CI on Linux it is the other way round.
 */
const status = await dawn();

for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  describe.skipIf(status.kind !== "ready" || f === null)(
    `bonded kernels on Dawn for ${name}`,
    () => {
      it("matches OpenMM's energies and our reference's forces", async () => {
        const report = await validateBonded(status.kind === "ready" ? status.handles.device : (null as never), {
          name,
          systemXml: readFileSync(`fixtures/${name}/system.xml`, "utf8"),
          stateXml: readFileSync(`fixtures/${name}/state.xml`, "utf8"),
          factsJson: readFileSync(`fixtures/${name}/facts.json`, "utf8"),
        });
        const failures = report.comparisons.filter((c) => !c.ok);
        expect(JSON.stringify(failures, null, 1)).toBe("[]");
        expect(report.comparisons).toHaveLength(3);
      });
    },
  );
}
