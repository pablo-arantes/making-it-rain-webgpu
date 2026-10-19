import { describe, expect, it } from "vitest";
import {
  exactFixedSum,
  f32Sum,
  FIXED_MAX,
  fixedWords,
  fromFixed,
  fromWords,
  toFixed,
} from "../src/reference/fixed-point.js";

/**
 * The format's resolution: one unit in the last place of the low word. Rounding
 * to nearest bounds the conversion error at half of it.
 */
const RESOLUTION = 2 ** -32;

describe("fixed-point representation", () => {
  it("round-trips a value whose bits all fit the format", () => {
    // Anything with no mantissa bits past 2^-32 is held exactly.
    for (const value of [0, 1, -1, 0.5, -0.5, 2.5, -2.5, 12345.5, -98765.25, 2 ** 30]) {
      const x = Math.fround(value);
      expect(fromFixed(toFixed(x))).toBe(x);
    }
  });

  it("round-trips everything else to within the format's resolution", () => {
    // An f32 below about 2^-9 has bits past 2^-32, which 32 fractional bits
    // cannot hold. That loss is 2.3e-10 kJ/mol/nm and is the whole of it.
    let seed = 1;
    const random = (): number => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    let worst = 0;
    for (let i = 0; i < 20000; i++) {
      const x = Math.fround((random() * 2 - 1) * 10 ** (random() * 8 - 2));
      const error = Math.abs(fromFixed(toFixed(x)) - x);
      expect(error).toBeLessThanOrEqual(RESOLUTION / 2);
      worst = Math.max(worst, error);
    }
    expect(worst).toBeLessThanOrEqual(RESOLUTION / 2);
  });

  it("is exactly symmetric in sign", () => {
    // The reason the conversion decomposes a magnitude rather than flooring a
    // signed value. Flooring makes a small negative value's fraction `1 + x`,
    // which an f32 near 1 cannot hold the low bits of.
    const x = Math.fround(-1e-9);
    expect(Math.fround(x - Math.floor(x))).toBe(1); // what we avoid doing
    expect(toFixed(x)).toBe(-toFixed(-x));

    let seed = 3;
    const random = (): number => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    for (let i = 0; i < 20000; i++) {
      const v = Math.fround((random() * 2 - 1) * 10 ** (random() * 8 - 4));
      expect(toFixed(v)).toBe(-toFixed(-v));
    }
  });

  it("reads the high word back signed and the low word unsigned", () => {
    expect(fixedWords(2.5)).toEqual({ lo: 2 ** 31, hi: 2 });
    expect(fixedWords(-2.5)).toEqual({ lo: 2 ** 31, hi: -3 });
    expect(fromFixed(toFixed(-2.5))).toBe(-2.5);
  });

  it("refuses values outside the representable range", () => {
    expect(() => toFixed(FIXED_MAX)).toThrow(/outside the fixed-point range/);
    expect(() => toFixed(-FIXED_MAX)).toThrow(/outside the fixed-point range/);
    expect(() => toFixed(Number.NaN)).toThrow(/not finite/);
    expect(() => toFixed(Number.POSITIVE_INFINITY)).toThrow(/not finite/);
  });

  it("reassembles the two words the GPU writes, with an unsigned low word", () => {
    // 0xFFFFFFFF as an i32 is -1; as the low word it must read as 4294967295.
    expect(fromWords(-1, 0)).toBe(4294967295n);
    expect(fromWords(0, -1)).toBe(-4294967296n);
    expect(fromFixed(fromWords(2 ** 31, 2))).toBe(2.5);
  });
});

describe("exact summation", () => {
  it("is order-independent, which is the whole point", () => {
    let seed = 7;
    const random = (): number => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const values = Array.from({ length: 5000 }, () =>
      Math.fround((random() * 2 - 1) * 10 ** (random() * 6 - 1)),
    );
    const forward = exactFixedSum(values);
    const backward = exactFixedSum([...values].reverse());
    const shuffled = exactFixedSum(
      [...values].sort(() => (random() < 0.5 ? -1 : 1)),
    );
    expect(backward).toBe(forward);
    expect(shuffled).toBe(forward);
  });

  it("recovers a sum that f32 accumulation loses", () => {
    // 2^24 is where f32 stops being able to add 1.
    const values = [Math.fround(2 ** 24), ...Array.from({ length: 64 }, () => 1)];
    expect(fromFixed(exactFixedSum(values))).toBe(2 ** 24 + 64);
    expect(f32Sum(values)).toBe(2 ** 24); // every 1 was swallowed
  });

  it("sums the cancelling magnitudes PME actually produces", () => {
    // The 3HTB system's direct and reciprocal space energies, which cancel to
    // about a tenth of either term.
    const direct = Math.fround(1563444.2152831473);
    const reciprocal = Math.fround(-1732547.546258145);
    const summed = fromFixed(exactFixedSum([direct, reciprocal]));
    // Both terms are above 2^-9, so the format holds them exactly and the
    // cancellation loses nothing at all.
    expect(summed).toBe(direct + reciprocal);
    expect(Math.abs(summed)).toBeGreaterThan(169000);
  });
});
