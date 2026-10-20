/**
 * The analysis page, driven as a user drives it.
 *
 * The analyses are verified against NumPy and MDTraj in Node
 * (`test/analysis-protein.test.ts`), so what only this can check is the
 * wiring: that a binary DCD arriving through a real file input is parsed, that
 * every figure draws, that the numbers on the page are the ones the headless
 * tests measured, and that the CSV and PNG downloads are real files.
 *
 * It also checks the guard that matters most here. A DCD carries coordinates
 * and nothing else, so atoms are matched to names by index; a topology with a
 * different atom count is not "close enough", it attaches every name to the
 * wrong coordinate. The page must refuse rather than analyse.
 */

import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";

const dcd = "fixtures/analysis/trajectory.dcd";
const pdb = "fixtures/1L2Y/system.pdb";
const reference = "fixtures/analysis/reference.json";
const present = existsSync(dcd) && existsSync(pdb) && existsSync(reference);

test.describe("the analysis page", () => {
  test.skip(!present, `${dcd} not generated — see tools/README.md`);

  test("is reachable, and analyses a trajectory end to end", async ({ page }) => {
    const failures: string[] = [];
    page.on("pageerror", (error) => failures.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") failures.push(message.text());
    });

    // Reachable from the front page, not only by URL.
    await page.goto("/");
    await page.locator('nav a[href="./analyse.html"]').click();
    await expect(page.locator("h1")).toHaveText("Analyse a trajectory");
    await page.waitForFunction(() => window.__mirAnalyse?.state === "idle");
    await expect(page.locator("#analyse")).toBeDisabled();

    await page.locator("#file-dcd").setInputFiles(dcd);
    await expect(page.locator("#trajectory-summary")).toContainText("60");
    // Still disabled with no topology: the Cα atoms cannot be found without it.
    await expect(page.locator("#analyse")).toBeDisabled();

    await page.locator("#file-top").setInputFiles(pdb);
    await expect(page.locator("#trajectory-summary")).toContainText("Cα");
    await expect(page.locator("#analyse")).toBeEnabled();

    await page.locator("#analyse").click();
    await page.waitForFunction(
      () =>
        window.__mirAnalyse?.state === "done" ||
        window.__mirAnalyse?.state === "failed",
      null,
      { timeout: 120_000 },
    );
    const result = await page.evaluate(() => window.__mirAnalyse);
    expect(result?.error ?? "").toBe("");
    expect(result?.state).toBe("done");

    // The same numbers test/analysis-protein.test.ts checked against NumPy.
    const expected = JSON.parse(readFileSync(reference, "utf8")) as {
      rmsd: number[];
      radiusOfGyration: number[];
      rmsf: number[];
    };
    const summary = result?.summary ?? {};
    expect(summary["frames"]).toBe(60);
    expect(summary["caAtoms"]).toBe(20);
    expect(Number(summary["rmsdMax"])).toBeCloseTo(Math.max(...expected.rmsd), 5);
    expect(Number(summary["rgMean"])).toBeCloseTo(
      expected.radiusOfGyration.reduce((a, b) => a + b, 0) / 60,
      5,
    );
    expect(Number(summary["rmsfMax"])).toBeCloseTo(Math.max(...expected.rmsf), 5);
    expect(Number(summary["pc1Fraction"])).toBeGreaterThan(0);
    expect(Number(summary["pc1Fraction"])).toBeLessThanOrEqual(100);

    // Every canvas drew something. A blank canvas is all zero alpha, so this
    // distinguishes "drawn" from "present in the markup".
    for (const id of [
      "plot-rmsd", "plot-rmsd-kde", "plot-rg", "plot-rg-kde",
      "plot-rmsf", "plot-2d", "plot-pca", "plot-correl",
    ]) {
      const painted = await page.evaluate((canvasId) => {
        const canvas = document.getElementById(canvasId) as HTMLCanvasElement;
        const context = canvas.getContext("2d");
        if (context === null) return 0;
        const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
        let opaque = 0;
        for (let i = 3; i < data.length; i += 4) if (data[i]! > 0) opaque++;
        return opaque;
      }, id);
      expect(painted, `#${id} drew nothing`).toBeGreaterThan(100);
    }

    await expect(page.locator("#pca-summary")).toContainText("PC1 carries");
    await expect(page.locator("#scale-2d")).toContainText("nm");

    const artefacts = (result?.artefacts ?? []).map((a) => a.name);
    expect(artefacts).toContain("analysis.csv");
    expect(artefacts).toContain("rmsd_2d.csv");
    expect(artefacts).toContain("cross_correlation.png");

    // A CSV download, parsed back: the header and one row of real numbers.
    const download = page.waitForEvent("download");
    await page.locator('[data-save="analysis.csv"]').click();
    const csv = readFileSync((await (await download).path())!, "utf8");
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("time_ps,rmsd_ca_nm,radius_of_gyration_nm,pc1,pc2");
    expect(lines).toHaveLength(61);
    const row = lines[1]!.split(",").map(Number);
    expect(row[0]).toBe(0);
    expect(row[1]).toBeCloseTo(0, 6);
    expect(row[2]).toBeCloseTo(expected.radiusOfGyration[0]!, 5);

    // And a PNG, which must be a real image rather than an empty data URL.
    const png = page.waitForEvent("download");
    await page.locator('[data-save="rmsd.png"]').click();
    const bytes = readFileSync((await (await png).path())!);
    expect(bytes.length).toBeGreaterThan(2000);
    expect([...bytes.subarray(0, 4)]).toEqual([0x89, 0x50, 0x4e, 0x47]);

    expect(failures, `console errors: ${failures.join(" | ")}`).toEqual([]);
    console.log(
      `\n  analysed ${summary["frames"]} frames, ${summary["caAtoms"]} Cα, ` +
        `PC1 ${summary["pc1Fraction"]}% of the motion`,
    );
  });

  test("refuses a topology that cannot be the same system", async ({ page }) => {
    // 🔴 The guard worth testing. Coordinates are matched to atoms by index,
    // so a mismatched atom count does not mean "approximately right" -- the Cα
    // selection would pick arbitrary atoms and every figure would be wrong
    // while looking entirely plausible.
    await page.goto("/analyse.html");
    await page.waitForFunction(() => window.__mirAnalyse?.state === "idle");
    await page.locator("#file-dcd").setInputFiles(dcd);
    await page.locator("#file-top").setInputFiles("fixtures/prep/1L2Y.pdb");
    await expect(page.locator("#trajectory-summary")).toContainText(
      "cannot be the same system",
    );
    await expect(page.locator("#analyse")).toBeDisabled();
  });
});
