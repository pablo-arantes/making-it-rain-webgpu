import { describe, expect, it } from "vitest";
import { erf, erfc, erfcApproximate, erfDerivative } from "../src/reference/erf.js";

/**
 * Reference values from mpmath at 30 digits, rounded to the nearest double.
 * External on purpose: the reference erfc is the yardstick the f32 kernel
 * approximation is measured against, so it is pinned against something that is
 * not this file.
 *
 * Writing a table from memory is how this test first "failed": the entry for
 * alpha*cutoff was a digit-shifted erf(3), and the implementation was right.
 */
const KNOWN: ReadonlyArray<readonly [x: number, erf: number, erfc: number]> = [
  [0, 0.0, 1.0],
  [0.1, 0.1124629160182849, 0.8875370839817152],
  [0.5, 0.5204998778130465, 0.4795001221869535],
  [1, 0.8427007929497149, 0.15729920705028513],
  [1.5, 0.9661051464753108, 0.033894853524689274],
  [2, 0.9953222650189527, 0.004677734981047266],
  // alpha * cutoff for the notebook's settings: the largest argument the
  // real-space Ewald sum ever evaluates.
  [2.628260884878466, 0.9997983354812846, 0.0002016645187154009],
  [3, 0.9999779095030014, 2.209049699858544e-5],
  [4, 0.9999999845827421, 1.541725790028002e-8],
];

describe("erf and erfc in double precision", () => {
  it("matches mpmath's erf to 14 digits", () => {
    for (const [x, want] of KNOWN) {
      expect(Math.abs(erf(x) - want)).toBeLessThan(1e-14);
      expect(Math.abs(erf(-x) + want)).toBeLessThan(1e-14);
    }
  });

  it("gives erfc to full relative precision, including where it is tiny", () => {
    for (const [x, , want] of KNOWN) {
      // Relative, not absolute: erfc(3) is 2.2e-5 and erfc(4) is 1.5e-8, and
      // the Ewald sum cares about them as fractions of themselves.
      expect(Math.abs((erfc(x) - want) / want)).toBeLessThan(1e-13);
      expect(Math.abs((erfc(-x) - (2 - want)) / (2 - want))).toBeLessThan(1e-13);
    }
    expect(erfc(0)).toBe(1);
    expect(erfc(Number.POSITIVE_INFINITY)).toBe(0);
    expect(erfc(Number.NEGATIVE_INFINITY)).toBe(2);
  });

  it("keeps erf(x) + erfc(x) = 1 across the range", () => {
    for (let x = -4; x <= 4; x += 0.05) {
      expect(Math.abs(erf(x) + erfc(x) - 1)).toBeLessThan(2e-15);
    }
  });

  it("has a derivative consistent with a central difference", () => {
    // h = 1e-4 rather than 1e-6. The error floor here is set by cancellation:
    // erf(x+h) - erf(x-h) is about 2h f'(x), and each erf carries ~1e-16
    // absolute, so the quotient inherits ~1e-16/(2h) relative. At h = 1e-6
    // that is 5e-11 in principle and 1.5e-8 in practice; at h = 1e-4 it is a
    // hundred times smaller, while the h^2 truncation term stays below 1e-7
    // over this range.
    const h = 1e-4;
    for (let x = 0.05; x < 3; x += 0.1) {
      const numerical = (erf(x + h) - erf(x - h)) / (2 * h);
      expect(Math.abs(erfDerivative(x) - numerical) / erfDerivative(x)).toBeLessThan(1e-6);
    }
  });
});

describe("the approximation the kernel uses", () => {
  it("stays within 1.2e-7 relative of the exact erfc over the Ewald range", () => {
    // 0 to alpha*cutoff is what the direct-space sum ever evaluates.
    let worst = 0;
    let worstAt = 0;
    for (let x = 0; x <= 2.7; x += 0.001) {
      const exact = erfc(x);
      const relative = Math.abs((erfcApproximate(x) - exact) / exact);
      if (relative > worst) {
        worst = relative;
        worstAt = x;
      }
    }
    expect(worst, `worst at x = ${worstAt}`).toBeLessThan(1.2e-7);
  });

  it("is still relatively accurate far beyond the cutoff", () => {
    for (const x of [3, 4, 5, 6]) {
      expect(Math.abs((erfcApproximate(x) - erfc(x)) / erfc(x))).toBeLessThan(1.2e-7);
    }
  });
});
