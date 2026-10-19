import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { checksum } from "../../src/analysis/npy.js";

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
    await page.waitForFunction(() => window.__mirLoop !== undefined, null, {
      timeout: 900_000,
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

    const cells = await page.evaluate(() => window.__mirCellList);
    if (cells !== undefined && "error" in cells) throw new Error(cells.error);
    // Bit-identical forces from a differently ordered list, which is what makes
    // the non-deterministic scatter safe. Asserted exactly, not approximately.
    expect(cells!.forceDifference).toBe(0);
    expect(cells!.deviceEnergy).toBe(cells!.hostEnergy);
    expect(JSON.stringify(cells, null, 1)).toContain('"ok": true');

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

    // The page fetched its trajectory fixtures over HTTP; this process can read
    // the same files off disk. Comparing checksums is what distinguishes a
    // kernel that computes the wrong answer from a page that was handed the
    // wrong input -- a distinction worth having, because one run disagreed with
    // OpenMM by 0.0167 nm while still matching the reference to 1.1e-7, which
    // is the signature of a shared bad input and not of a bad kernel.
    for (const [key, reported] of Object.entries(integrators!.checksums)) {
      const [label, kind] = key.split("/");
      const path = `fixtures/${name}/trajectory-${label}-${kind}.npy`;
      const bytes = readFileSync(path);
      const onDisk = checksum(
        bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      );
      expect(reported, `${path} was corrupted in transit to the page`).toBe(onDisk);
    }
    const failedSteps = integrators!.steps.filter((s) => !s.ok);
    expect(JSON.stringify(failedSteps, null, 1)).toBe("[]");
    // Reported separately from `ok` so a thermostat or reduction failure is not
    // read as an integrator failure: they fail for different reasons.
    expect(JSON.stringify(integrators!.thermostat, null, 1)).toContain('"ok": true');
    expect(JSON.stringify(integrators!.reductions, null, 1)).toContain('"ok": true');
    expect(integrators!.ok).toBe(true);

    const loop = await page.evaluate(() => window.__mirLoop);
    if (loop !== undefined && "error" in loop) throw new Error(loop.error);
    // Exactly zero: two runs of the same input differing at all is a race, not
    // a precision question.
    expect(loop!.determinismDifference).toBe(0);
    expect(loop!.onePositionVsReference).toBeLessThan(loop!.thresholds.onePosition);
    expect(loop!.driftPerAtomPerNs).toBeLessThan(loop!.thresholds.driftPerAtomPerNs);
    expect(loop!.ok).toBe(true);

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
        `\n  cell list              ${cells!.divisions.join("x")} = ${cells!.cellCount} cells` +
        `  ${cells!.assignmentDifferences} assignment diffs` +
        `  ${cells!.orderDifferences} reordered` +
        `  force difference ${cells!.forceDifference}` +
        `\n  generator              ${randomness!.bitMismatches} bit mismatches` +
        `  worst deviate ${randomness!.worstDeviate.toExponential(2)}` +
        `  at radius ${randomness!.worstRadius.toExponential(2)}` +
        `\n  fixtures               ${Object.entries(integrators!.checksums)
          .map(([k, v]) => `${k}=${v.toString(16)}`)
          .join(" ")}` +
        `\n  reductions             COM ${integrators!.reductions.comVelocityError.toExponential(2)} nm/ps` +
        `  kinetic ${integrators!.reductions.kineticEnergy.toFixed(2)} kJ/mol` +
        `  rel ${integrators!.reductions.kineticEnergyRelativeError.toExponential(2)}` +
        `\n  loop                   ${loop!.passesPerStep} passes/step` +
        `  determinism ${loop!.determinismDifference}` +
        `  one step vs ref ${loop!.onePositionVsReference.toExponential(2)} nm` +
        `\n  conservation           ${loop!.conservationSteps} steps` +
        ` (${loop!.conservationPicoseconds.toFixed(1)} ps)` +
        `  drift ${loop!.driftPerAtomPerNs.toFixed(4)} kJ/mol/atom/ns` +
        `  ${loop!.firstReport.temperature.toFixed(1)}→${loop!.lastReport.temperature.toFixed(1)} K` +
        `\n  throughput             ${loop!.nsPerDay.toFixed(2)} ns/day at 2 fs` +
        `  peak pool ${(loop!.peakPoolBytes / (1024 * 1024)).toFixed(1)} MB`,
    );
  });
}
