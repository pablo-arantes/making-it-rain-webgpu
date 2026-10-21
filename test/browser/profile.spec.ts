/**
 * The per-kernel timing breakdown, in Chrome, on whatever GPU is here.
 *
 * This lane exists because the breakdown is the only thing that told us where a
 * step's time actually went, and two of the three things it found were in
 * places reasoning had already cleared. So the breakdown itself has to keep
 * working: a profile that silently reports nothing, or that attributes a step
 * to passes that are not the ones encoded, is worse than no profile, because
 * the next person optimises whatever it names.
 *
 * It asserts shape and self-consistency, not a speed. A speed assertion would
 * fail on a slower GPU than the one it was written on, which is every machine
 * this is meant to run on; `docs/measurements/profile-step-m3pro.json` is where
 * numbers live. What is checked is that every pass of a step is accounted for,
 * that the shares are a distribution, and that the GPU time the query set
 * reports is consistent with the wall-clock the caller actually waited --
 * which is what catches a timestamp that is being read in the wrong units or
 * from the wrong pass.
 */

import { expect, test } from "@playwright/test";
import { existsSync } from "node:fs";

const fixture = "fixtures/1L2Y/system.xml";
const present = existsSync(fixture);

test.describe("the step profile", () => {
  test.skip(!present, `${fixture} not generated — see tools/README.md`);

  test("accounts for every pass of a step, and agrees with the clock", async ({
    page,
  }) => {
    const failures: string[] = [];
    page.on("pageerror", (error) => failures.push(String(error)));

    await page.goto("/validate.html?fixture=1L2Y");
    await page.waitForFunction(() => window.__mirProfile !== undefined, null, {
      timeout: 600_000,
    });
    const profile = (await page.evaluate(() => window.__mirProfile))!;
    expect(failures).toEqual([]);

    // `timestamp-query` is optional, and a device without it must say so rather
    // than report a profile full of zeroes.
    if (!profile.available) {
      expect(profile.reason).toBeTruthy();
      test.info().annotations.push({
        type: "skipped",
        description: `no timestamp-query: ${profile.reason}`,
      });
      return;
    }

    expect(profile.particleCount).toBe(2373);
    expect(profile.repeats).toBeGreaterThan(1);

    // The passes a step encodes: the cell list including its fold, the forces,
    // and the integrator. Naming them is the point -- an empty or near-empty
    // list is exactly the failure this test exists for.
    const labels = profile.kernels.map((k) => k.label);
    for (const expected of [
      "cell/wrap",
      "cell/assign",
      "direct space pairs",
      "pme spread",
      "pme convolve",
      "pme interpolate",
      "HarmonicBondForce evaluate",
      "HarmonicAngleForce evaluate",
      "PeriodicTorsionForce evaluate",
    ]) {
      expect(labels, `the profile should name ${expected}`).toContain(expected);
    }
    expect(new Set(labels).size).toBe(labels.length);

    // The shares are a distribution over the passes, ordered worst first.
    const shares = profile.kernels.map((k) => k.share);
    expect(shares.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    for (const share of shares) expect(share).toBeGreaterThan(0);
    expect([...shares].sort((a, b) => b - a)).toEqual(shares);

    // Every pass's own time sums to the step's GPU time.
    const summed = profile.kernels.reduce((a, k) => a + k.microseconds, 0);
    expect(summed).toBeCloseTo(profile.perStepGpuMicroseconds, 3);

    // The kernels cannot have taken longer than the caller waited, and the gap
    // is launch overhead rather than a unit error. A timestamp read in the
    // wrong units lands orders of magnitude outside this.
    expect(profile.perStepGpuMicroseconds).toBeLessThanOrEqual(
      profile.perStepWallMicroseconds * 1.05,
    );
    expect(profile.overheadShare).toBeGreaterThanOrEqual(-0.05);
    expect(profile.overheadShare).toBeLessThan(0.9);
    expect(profile.impliedStepsPerSecond).toBeCloseTo(
      1e6 / profile.perStepWallMicroseconds,
      3,
    );

    // The pair kernel is the one worth knowing about: it is the step on every
    // device measured so far, and the reason the shell is split across the
    // dispatch and the minimum image is loop-free.
    const pairs = profile.kernels.find((k) => k.label === "direct space pairs")!;
    expect(pairs.share).toBeGreaterThan(0.2);

    // And the page says so, so a person loading it sees the same numbers.
    await expect(page.locator("h3", { hasText: "Where a step's time goes" })).toBeVisible();
    await expect(page.locator("td code", { hasText: "direct space pairs" })).toBeVisible();
  });
});
