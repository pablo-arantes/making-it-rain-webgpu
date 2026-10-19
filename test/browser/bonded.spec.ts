import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";

/**
 * The bonded kernels against OpenMM, in a browser, on both fixtures.
 *
 * This is the lane that actually exercises the WGSL on macOS, since no
 * published Dawn binary loads here (CLAUDE.md). It drives the validation page,
 * which fetches the same `system.xml` the Python oracle wrote.
 */
for (const name of ["1L2Y", "3HTB"]) {
  const present = existsSync(`fixtures/${name}/system.xml`);
  test.skip(!present, `fixtures/${name} not generated — see tools/README.md`);

  test(`forces match OpenMM for ${name}`, async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (m) => {
      if (m.type() === "error") consoleErrors.push(m.text());
    });
    page.on("pageerror", (e) => consoleErrors.push(String(e)));

    await page.goto(`/validate.html?fixture=${name}`);
    await page.waitForFunction(() => window.__mirValidation !== undefined, null, {
      timeout: 300_000,
    });
    const report = await page.evaluate(() => window.__mirValidation);
    if (report !== undefined && "error" in report) throw new Error(report.error);
    const checked = report!;

    // Every force the system carries must be present and must pass.
    expect(checked.comparisons.map((c) => c.forceType).sort()).toEqual([
      "HarmonicAngleForce",
      "HarmonicBondForce",
      "NonbondedForce[direct]",
      "PeriodicTorsionForce",
    ]);
    const failures = checked.comparisons.filter((c) => !c.ok);
    expect(JSON.stringify(failures, null, 1)).toBe("[]");
    expect(checked.ok).toBe(true);
    expect(consoleErrors).toEqual([]);

    console.log(
      `\n${name}: ${checked.particleCount} particles\n` +
        checked.comparisons
          .map(
            (c) =>
              `  ${c.forceType.padEnd(21)} ${c.termCount.toString().padStart(6)} terms` +
              `  energy ${c.deviceEnergy.toFixed(4).padStart(14)}` +
              `  vs OpenMM ${c.energyVsOpenmmRelativeError?.toExponential(2) ?? "—"}` +
              `  force rel RMS ${c.forceRelativeRms.toExponential(2)}`,
          )
          .join("\n"),
    );
  });
}
