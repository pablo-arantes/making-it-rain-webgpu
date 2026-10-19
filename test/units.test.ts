import { describe, expect, it } from "vitest";
import {
  BOLTZMANN_KJ_PER_MOL_K,
  ewaldAlpha,
  nmFromAngstrom,
  ONE_4PI_EPS0,
  pmeMinimumGridSize,
  psFromFs,
  smoothFftSize,
} from "../src/engine/units.js";
import facts3HTB from "../docs/measurements/3HTB-JZ4-facts.json";
import facts1L2Y from "../docs/measurements/1L2Y-facts.json";

/**
 * The committed measurements are the fixtures.  If OpenMM's own numbers and
 * ours ever disagree, this is where it shows, and the answer is to fix the
 * code -- never to edit the measurement.
 */
describe("PME parameters, against OpenMM's own output", () => {
  const CUTOFF_NM = 1.0;
  const TOLERANCE = 0.0005;

  it("derives the same alpha OpenMM derives", () => {
    const alpha = ewaldAlpha(CUTOFF_NM, TOLERANCE);
    expect(alpha).toBeCloseTo(facts3HTB.pme.alpha_per_nm, 12);
    // Every system shares it, because it depends only on cutoff and tolerance.
    expect(alpha).toBeCloseTo(facts1L2Y.pme.alpha_per_nm, 12);
  });

  it("derives the same minimum grid the Reference platform used", () => {
    for (const facts of [facts3HTB, facts1L2Y]) {
      const boxNm = facts.box_nm[0]![0]!;
      const minimum = pmeMinimumGridSize(
        ewaldAlpha(CUTOFF_NM, TOLERANCE),
        boxNm,
        TOLERANCE,
      );
      expect(minimum).toBe(facts.pme.grid_reference_platform[0]);
    }
  });

  it("never returns a grid below OpenMM's floor of 6", () => {
    expect(pmeMinimumGridSize(ewaldAlpha(1, 0.0005), 0.1, 0.0005)).toBe(6);
  });
});

describe("smoothFftSize", () => {
  it("rounds 3HTB's minimum of 53 up to a 2/3/5/7-smooth 54", () => {
    expect(smoothFftSize(53)).toBe(54);
  });

  it("leaves an already-smooth size alone", () => {
    for (const n of [25, 32, 48, 54, 64, 96]) expect(smoothFftSize(n)).toBe(n);
  });

  it("only ever returns 2/3/5/7-smooth sizes", () => {
    for (let n = 1; n <= 200; n++) {
      let r = smoothFftSize(n);
      expect(r).toBeGreaterThanOrEqual(n);
      for (const p of [2, 3, 5, 7]) while (r % p === 0) r /= p;
      expect(r).toBe(1);
    }
  });
});

describe("constants and conversions", () => {
  it("uses OpenMM's values", () => {
    // kT at 298 K, the notebook's temperature.
    expect(BOLTZMANN_KJ_PER_MOL_K * 298).toBeCloseTo(2.477709860209666, 12);
    expect(ONE_4PI_EPS0).toBeCloseTo(138.93545764498457, 12);
  });

  it("converts the notebook's UI units once, at the boundary", () => {
    expect(nmFromAngstrom(4)).toBeCloseTo(0.4, 15); // the padding default
    expect(psFromFs(2)).toBeCloseTo(0.002, 15); // the timestep default
  });
});
