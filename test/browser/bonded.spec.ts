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
    await page.waitForFunction(() => window.__mirIntegrators !== undefined, null, {
      timeout: 600_000,
    });
    const report = await page.evaluate(() => window.__mirValidation);
    if (report !== undefined && "error" in report) throw new Error(report.error);
    const checked = report!;

    // Every force the system carries must be present and must pass.
    expect(checked.comparisons.map((c) => c.forceType).sort()).toEqual([
      "HarmonicAngleForce",
      "HarmonicBondForce",
      "NonbondedForce[direct]",
      "NonbondedForce[reciprocal]",
      "PeriodicTorsionForce",
    ]);
    const failures = checked.comparisons.filter((c) => !c.ok);
    expect(JSON.stringify(failures, null, 1)).toBe("[]");
    expect(checked.ok).toBe(true);
    expect(consoleErrors).toEqual([]);

    const constraints = await page.evaluate(() => window.__mirConstraints);
    if (constraints !== undefined && "error" in constraints) {
      throw new Error(constraints.error);
    }
    expect(JSON.stringify(constraints, null, 1)).toContain('"ok": true');

    const randomness = await page.evaluate(() => window.__mirRandom);
    if (randomness !== undefined && "error" in randomness) {
      throw new Error(randomness.error);
    }
    // The integer stream has to be exact; the deviates only close.
    expect(randomness!.bitMismatches).toBe(0);
    expect(randomness!.worstDeviate).toBeLessThan(randomness!.thresholds.deviate);

    const integrators = await page.evaluate(() => window.__mirIntegrators);
    if (integrators !== undefined && "error" in integrators) {
      throw new Error(integrators.error);
    }
    const failedSteps = integrators!.steps.filter((s) => !s.ok);
    expect(JSON.stringify(failedSteps, null, 1)).toBe("[]");
    // Reported separately from `ok` so a thermostat or reduction failure is not
    // read as an integrator failure: they fail for different reasons.
    expect(JSON.stringify(integrators!.thermostat, null, 1)).toContain('"ok": true');
    expect(JSON.stringify(integrators!.reductions, null, 1)).toContain('"ok": true');
    expect(integrators!.ok).toBe(true);

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
          .join("\n") +
        `\n  constraints            ${constraints!.settleClusters} SETTLE + ` +
        `${constraints!.iterativeClusters} iterative` +
        `  vs reference ${constraints!.worstDifference.toExponential(2)} nm` +
        `  constraint error ${constraints!.deviceWorstRelativeError.toExponential(2)}` +
        integrators!.steps
          .map(
            (s) =>
              `\n  ${s.integrator.padEnd(22)} ${s.steps} steps` +
              `  position vs ref ${s.worstPositionVsReference.toExponential(2)} nm` +
              `  vs OpenMM ${s.worstPositionVsOpenmm.toExponential(2)} nm` +
              `  velocity vs ref ${s.worstVelocityVsReference.toExponential(2)}`,
          )
          .join("") +
        `\n  thermostat             friction ${integrators!.thermostat.friction}` +
        `  position vs ref ${integrators!.thermostat.worstPositionVsReference.toExponential(2)} nm` +
        `  velocity vs ref ${integrators!.thermostat.worstVelocityVsReference.toExponential(2)}` +
        `  (moved ${integrators!.thermostat.velocityChange.toFixed(3)})` +
        // The stages, because an end-to-end number measures their composition
        // and can be far better conditioned than any one of them. See
        // docs/DYNAMICS.md.
        `\n  thermostat stages      kick ${integrators!.thermostat.stages.afterKick.toExponential(2)}` +
        `  rattle ${integrators!.thermostat.stages.afterRattle.toExponential(2)}` +
        `  drift ${integrators!.thermostat.stages.afterDriftVelocity.toExponential(2)}` +
        `/${integrators!.thermostat.stages.afterDriftPosition.toExponential(2)}` +
        `  shake ${integrators!.thermostat.stages.afterShake.toExponential(2)}` +
        `  finish ${integrators!.thermostat.stages.afterFinishVelocity.toExponential(2)}` +
        `\n  generator              ${randomness!.bitMismatches} bit mismatches` +
        `  worst deviate ${randomness!.worstDeviate.toExponential(2)}` +
        `  at radius ${randomness!.worstRadius.toExponential(2)}` +
        `\n  reductions             COM ${integrators!.reductions.comVelocityError.toExponential(2)} nm/ps` +
        `  kinetic ${integrators!.reductions.kineticEnergy.toFixed(2)} kJ/mol` +
        `  rel ${integrators!.reductions.kineticEnergyRelativeError.toExponential(2)}`,
    );
  });
}
