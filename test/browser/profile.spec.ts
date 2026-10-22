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
      // The neighbour-list path, which is what carries the force. The
      // cell-shell pass is encoded too but handles only overflowed atoms, so
      // on a healthy run its span is too short to measure and the profiler
      // drops it rather than reporting a zero -- requiring it here would be
      // requiring an overflow.
      "direct space pairs (list)",
      "verlet/count",
      "verlet/fill",
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

    // The pair work is the one worth knowing about: it is the step on every
    // device measured so far, and the reason the shell is split across the
    // dispatch, the minimum image is loop-free and there is a neighbour list
    // at all. Counted with the list's build, which is pair work paid in
    // advance.
    const pairShare = profile.kernels
      .filter((k) => k.label.startsWith("direct space pairs") || k.label.startsWith("verlet/"))
      .reduce((total, k) => total + k.share, 0);
    expect(pairShare).toBeGreaterThan(0.2);

    // A run is faster than the path the list replaced, and ends in the same
    // place. The factor is not asserted -- it is 1.5x to 1.8x on the GPU this
    // was written on and will differ elsewhere -- but the direction is not a
    // matter of hardware, and the energies agreeing is what says the two runs
    // are the same physics rather than two different trajectories.
    expect(profile.steadyStepsPerSecond).toBeGreaterThan(profile.shellStepsPerSecond);
    expect(profile.listEnergyPerAtom).toBeCloseTo(profile.shellEnergyPerAtom, 6);
    expect(profile.overflowed).toBe(false);
    expect(profile.stepsPerRebuild).toBeGreaterThan(1);

    // And the page says so, so a person loading it sees the same numbers.
    await expect(page.locator("h3", { hasText: "Where a step's time goes" })).toBeVisible();
    await expect(
      page.locator("td code", { hasText: "direct space pairs (list)" }),
    ).toBeVisible();

    // The list is held to bit-for-bit agreement with the shell scan, which is
    // the check that a lost pair cannot hide behind a tolerance.
    const neighbours = (await page.evaluate(() => window.__mirNeighbours))!;
    expect("error" in neighbours ? neighbours.error : "").toBe("");
    if (!("error" in neighbours)) {
      expect(neighbours.differing).toBe(0);
      expect(neighbours.worstForceDifference).toBe(0);
      expect(neighbours.ok).toBe(true);
      // One cell has to be at least as wide as the list's radius, or the
      // 27-cell shell cannot reach the pairs the list is supposed to hold --
      // unless there are exactly three cells an axis, where the shell is the
      // whole box.
      const narrowest = Math.min(...neighbours.cellExtent);
      expect(
        narrowest >= neighbours.searchRadius || neighbours.cells.every((n) => n === 3),
      ).toBe(true);
    }
  });
});
