/**
 * The run page, driven as a user drives it.
 *
 * Everything underneath is tested elsewhere -- the protocol in Node against a
 * stub, the kernels against OpenMM, the artefacts against three trajectory
 * readers. What only this can check is that the page wires them together: that
 * uploading files through a real `<input type="file">` reaches the parser, that
 * "verify" reports on the user's own system, that starting a run fills the
 * plots and the progress text, and that the artefacts land in OPFS with
 * downloadable sizes.
 *
 * The 1 ns gate from docs/PLAN.md is the same page with bigger numbers, behind
 * `MIR_LONG_RUN=1`: at the measured 38 ns/day it is about 38 minutes, which is
 * not something a default lane should spend.
 */

import { expect, test } from "@playwright/test";
import { existsSync, mkdirSync } from "node:fs";

const fixture = "1L2Y";
const present = existsSync(`fixtures/${fixture}/system.xml`);

test.describe("the run page", () => {
  test.skip(!present, `fixtures/${fixture} not generated — see tools/README.md`);

  test("uploads, verifies, runs and writes artefacts", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (m) => {
      if (m.type() === "error") consoleErrors.push(m.text());
    });
    page.on("pageerror", (e) => consoleErrors.push(String(e)));

    await page.goto("/run.html");
    await page.waitForFunction(() => window.__mirRun?.state === "idle");
    // The page must find a GPU before anything else is meaningful.
    await expect(page.locator("#start")).toBeEnabled();

    // Upload through the real control, not by calling a function: the file
    // reading path is part of what is being tested.
    await page.locator("#file-system").setInputFiles(`fixtures/${fixture}/system.xml`);
    await page.locator("#file-state").setInputFiles(`fixtures/${fixture}/state.xml`);
    if (existsSync(`fixtures/${fixture}/system-restrained.xml`)) {
      await page
        .locator("#file-restrained")
        .setInputFiles(`fixtures/${fixture}/system-restrained.xml`);
    }
    await page.locator("#file-pdb").setInputFiles(`fixtures/${fixture}/system.pdb`);
    await expect(page.locator("#upload-summary")).toContainText("2,373");
    await expect(page.locator("#upload-summary")).toContainText("NonbondedForce");

    // --- Verify -----------------------------------------------------------
    await page.locator("#verify").click();
    await page.waitForFunction(() => window.__mirVerify !== undefined, null, {
      timeout: 600_000,
    });
    const verify = await page.evaluate(() => window.__mirVerify);
    if (verify !== undefined && "error" in verify) throw new Error(verify.error);
    expect(verify!.particleCount).toBe(2373);
    expect(verify!.moleculesIntact).toBe(true);
    expect(verify!.energyRelativeError).toBeLessThan(
      verify!.thresholds.energyRelative,
    );
    expect(verify!.forceRelativeRms).toBeLessThan(
      verify!.thresholds.forceRelativeRms,
    );
    // 1L2Y's box is 3.08 cutoffs wide, so it cannot take a barostat, and the
    // page has to say so rather than failing partway into a run.
    expect(verify!.canRunNpt).toBe(false);
    await expect(page.locator("#verify-body")).toContainText("constant volume");

    // --- A short run ------------------------------------------------------
    const unique = `pw${Date.now()}`;
    await page.locator("#job").fill(unique);
    await page.locator("#equil-ns").fill("0.002");
    await page.locator("#stride-ns").fill("0.002");
    await page.locator("#strides").fill("2");
    await page.locator("#traj-interval").fill("1");
    await page.locator("#log-interval").fill("1");
    await page.locator("#min-steps").fill("20");

    await page.locator("#start").click();
    await page.waitForFunction(
      () => window.__mirRun?.state === "done" || window.__mirRun?.state === "failed",
      null,
      { timeout: 900_000 },
    );
    const run = await page.evaluate(() => window.__mirRun);
    expect(run?.error ?? "").toBe("");
    expect(run?.state).toBe("done");

    // Three stages, four artefacts each, plus run.json.
    const artefacts = (run?.artefacts ?? []).map((a) => a.name).sort();
    expect(artefacts).toEqual(
      [
        "run.json",
        ...[`${unique}_equil`, `${unique}_prod_1`, `${unique}_prod_2`].flatMap(
          (base) => [`${base}.dcd`, `${base}.log`, `${base}.pdb`, `${base}.rst`],
        ),
      ].sort(),
    );
    for (const artefact of run?.artefacts ?? []) {
      expect(artefact.bytes, artefact.name).toBeGreaterThan(0);
    }

    // The things a user watches: reports arrived, the progress text was filled
    // in, and a download button exists for every artefact.
    expect(run?.reports ?? 0).toBeGreaterThan(0);
    await expect(page.locator("#run-progress-text")).toContainText("ns/day");
    await expect(page.locator("#run-status")).toContainText("Done");
    expect(
      await page.locator("#artefact-body button[data-download]").count(),
    ).toBe(artefacts.length);

    // --- Take the artefacts away ------------------------------------------
    // Downloading through the real button, and saving where
    // `tools/check_dcd.py` can be pointed at it: the plan's gate is that the
    // trajectories open in MDAnalysis, MDTraj, pytraj and VMD, and a 2,373-atom
    // DCD from an actual run is a much better subject for that than the
    // 37-atom synthetic one the writer's unit test uses.
    const saved = "fixtures/run-artefacts";
    mkdirSync(saved, { recursive: true });
    for (const suffix of ["dcd", "pdb", "rst", "log"]) {
      const name = `${unique}_equil.${suffix}`;
      const [download] = await Promise.all([
        page.waitForEvent("download"),
        page.locator(`#artefact-body button[data-download="${name}"]`).click(),
      ]);
      expect(download.suggestedFilename()).toBe(name);
      await download.saveAs(`${saved}/equil.${suffix}`);
    }
    expect(existsSync(`${saved}/equil.dcd`)).toBe(true);

    expect(consoleErrors.filter((m) => !m.includes("404"))).toEqual([]);
    console.log(
      `\n  run page: ${verify!.particleCount} particles, verify ` +
        `${verify!.energyRelativeError.toExponential(2)} energy / ` +
        `${verify!.forceRelativeRms.toExponential(2)} force in ` +
        `${verify!.elapsed.toFixed(1)} s, NPT ${verify!.canRunNpt}` +
        `\n  ${artefacts.length} artefacts, ${run?.reports} reports, ` +
        `${((run?.artefacts ?? []).reduce((n, a) => n + a.bytes, 0) / 1024).toFixed(1)} kB` +
        `\n  downloaded equil.{dcd,pdb,rst,log} to fixtures/run-artefacts/`,
    );
  });

  test("reports what this GPU can hold, before running anything", async ({ page }) => {
    // 🔴 Phase 5's cross-device work, in the only form a single-GPU machine can
    // check in a browser: that the real adapter's limits reach the fitness
    // logic and that the page says what they allow. The arithmetic itself is
    // tested against other devices' limits in test/device-fitness.test.ts,
    // because this machine has one GPU and the floor profile is the guarantee
    // that matters.
    await page.goto("/run.html");
    await page.waitForFunction(() => window.__mirRun?.state === "idle");
    await page.locator("#file-system").setInputFiles(`fixtures/${fixture}/system.xml`);
    await page.locator("#file-state").setInputFiles(`fixtures/${fixture}/state.xml`);

    // The ceiling is stated at upload, not discovered during a run.
    await expect(page.locator("#upload-summary")).toContainText("holds about");
    const summary = (await page.locator("#upload-summary").textContent()) ?? "";
    const held = Number(
      (summary.match(/holds about ([\d,]+) atoms/)?.[1] ?? "0").replace(/,/g, ""),
    );
    expect(held).toBeGreaterThan(100_000);
    // Whatever it reports has to exceed the system it just accepted.
    expect(held).toBeGreaterThan(2373);

    // And the verification carries the same verdict, so the two cannot drift.
    await page.locator("#verify").click();
    await page.waitForFunction(() => window.__mirVerify !== undefined, null, {
      timeout: 300_000,
    });
    const verify = await page.evaluate(() => window.__mirVerify);
    const fitness = (verify as { fitness?: { fits: boolean; maxAtoms: number } })
      ?.fitness;
    expect(fitness?.fits).toBe(true);
    expect(fitness?.maxAtoms).toBe(held);
  });

  test("runs a nanosecond end to end", async ({ page }) => {
    // The plan's Phase 2 gate. Opt-in: about 38 minutes at the measured
    // 38 ns/day for this fixture, which no default lane should spend.
    test.skip(process.env["MIR_LONG_RUN"] !== "1", "set MIR_LONG_RUN=1");
    test.setTimeout(5_400_000);

    await page.goto("/run.html");
    await page.waitForFunction(() => window.__mirRun?.state === "idle");
    await page.locator("#file-system").setInputFiles(`fixtures/${fixture}/system.xml`);
    await page.locator("#file-state").setInputFiles(`fixtures/${fixture}/state.xml`);
    await page.locator("#file-pdb").setInputFiles(`fixtures/${fixture}/system.pdb`);
    await page.locator("#job").fill("gate1ns");
    await page.locator("#equil-ns").fill("0.1");
    await page.locator("#stride-ns").fill("0.9");
    await page.locator("#strides").fill("1");
    await page.locator("#start").click();
    await page.waitForFunction(
      () => window.__mirRun?.state === "done" || window.__mirRun?.state === "failed",
      null,
      { timeout: 5_400_000 },
    );
    const run = await page.evaluate(() => window.__mirRun);
    expect(run?.error ?? "").toBe("");
    expect(run?.state).toBe("done");
    const bytes = (run?.artefacts ?? []).reduce((n, a) => n + a.bytes, 0);
    console.log(
      `\n  1 ns gate: ${run?.artefacts?.length} artefacts, ` +
        `${(bytes / 1048576).toFixed(1)} MB, ${run?.reports} reports`,
    );
    // A nanosecond at a 10 ps trajectory interval is 100 frames.
    expect(bytes).toBeGreaterThan(1_000_000);
  });
});
