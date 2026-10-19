/**
 * The counter-based generator, against Random123's own known answers.
 *
 * Three groups of checks, in order of how much they prove.
 *
 * The **known-answer vectors** come from Random123's `tests/kat_vectors` and
 * are the only evidence that this is Threefry and not merely something
 * Threefry-shaped. A key schedule off by one injection, or a rotation table
 * rotated by one position, gives a stream that passes every statistical test
 * below and is not the published generator.
 *
 * The **statistical checks** then confirm the parts the vectors cannot: that
 * the Box-Muller wrapper produces standard normals, and that the three deviates
 * one particle gets are independent of each other, of its neighbours' and of
 * the next step's. A thermostat whose noise correlates across particles heats
 * the system anisotropically, which is the kind of bug that shows up as a
 * plausible-but-wrong temperature days later.
 *
 * Bit equality with the WGSL twin is checked in the browser lane, where there
 * is a device to run it on (src/gpu/validate-random.ts). The uniforms match
 * exactly by construction; the deviates only to f32 precision, because WGSL
 * pins down `log` and `cos` to a few ulp and no further.
 */

import { describe, expect, it } from "vitest";
import {
  normalPair,
  openUnitInterval,
  threefry2x32,
  thermostatNormal,
  thermostatNormals,
} from "../src/reference/random.js";

const hex = (n: number): string => (n >>> 0).toString(16).padStart(8, "0");
const words = (text: string): number[] =>
  text.split(" ").map((h) => parseInt(h, 16));

/**
 * Verbatim from DEShawResearch/random123 `tests/kat_vectors`, the
 * `threefry2x32` rows: rounds, then counter and key, then the expected output.
 */
const KNOWN_ANSWERS: readonly (readonly [number, string, string])[] = [
  [13, "00000000 00000000 00000000 00000000", "9d1c5ec6 8bd50731"],
  [13, "ffffffff ffffffff ffffffff ffffffff", "fd36d048 2d17272c"],
  [13, "243f6a88 85a308d3 13198a2e 03707344", "ba3e4725 f27d669e"],
  [20, "00000000 00000000 00000000 00000000", "6b200159 99ba4efe"],
  [20, "ffffffff ffffffff ffffffff ffffffff", "1cb996fc bb002be7"],
  [20, "243f6a88 85a308d3 13198a2e 03707344", "c4923a9c 483df7a0"],
  [32, "00000000 00000000 00000000 00000000", "cee3d47e a23dfd5c"],
  [32, "ffffffff ffffffff ffffffff ffffffff", "6e2fe0d0 b1b76f82"],
  [32, "243f6a88 85a308d3 13198a2e 03707344", "e2827716 c3c05cdf"],
];

describe("threefry2x32", () => {
  it.each(KNOWN_ANSWERS)(
    "matches Random123's known answer at %i rounds for %s",
    (rounds, input, expected) => {
      const w = words(input);
      const out = threefry2x32(
        [w[0] as number, w[1] as number],
        [w[2] as number, w[3] as number],
        rounds,
      );
      expect(`${hex(out[0])} ${hex(out[1])}`).toBe(expected);
    },
  );

  it("is a bijection on the counter for a fixed key", () => {
    // Not a proof, but a collision here would mean the mixing is broken in a
    // way that quietly reuses deviates between particles.
    const seen = new Set<string>();
    for (let i = 0; i < 20000; i++) {
      const out = threefry2x32([i, 0], [12345, 0]);
      seen.add(`${out[0]},${out[1]}`);
    }
    expect(seen.size).toBe(20000);
  });

  it("changes every output bit about half the time when the counter moves", () => {
    // Avalanche: flipping one counter bit should flip each output bit with
    // probability 1/2. A missing key injection shows up here as a column stuck
    // near 0 or 1.
    const trials = 4000;
    const flips = new Array<number>(64).fill(0);
    for (let i = 0; i < trials; i++) {
      const a = threefry2x32([i, 7], [99, 0]);
      const b = threefry2x32([i ^ 1, 7], [99, 0]);
      for (let bit = 0; bit < 32; bit++) {
        if ((((a[0] ^ b[0]) >>> bit) & 1) === 1) flips[bit] = (flips[bit] as number) + 1;
        if ((((a[1] ^ b[1]) >>> bit) & 1) === 1)
          flips[32 + bit] = (flips[32 + bit] as number) + 1;
      }
    }
    for (const count of flips) {
      expect(Math.abs(count / trials - 0.5)).toBeLessThan(0.04);
    }
  });
});

describe("openUnitInterval", () => {
  it("never reaches either endpoint", () => {
    // Box-Muller takes a logarithm of this, so a zero is an infinity.
    expect(openUnitInterval(0)).toBeGreaterThan(0);
    expect(openUnitInterval(0xffffffff)).toBeLessThan(1);
    expect(openUnitInterval(0)).toBe(2 ** -24);
    expect(openUnitInterval(0xffffffff)).toBe(1 - 2 ** -24);
  });

  it("is exactly representable in f32, for every input", () => {
    // The whole reason it is 23 bits and not 24. An f64 host and an f32 device
    // have to produce the same uniform, and that only holds if the value is
    // representable in the narrower format.
    for (let bits = 0; bits < 0x20000; bits++) {
      const value = openUnitInterval(bits * 0x8003);
      expect(Math.fround(value)).toBe(value);
    }
    for (const bits of [0, 1, 0x1234_5678, 0x8000_0000, 0xffff_ffff]) {
      expect(Math.fround(openUnitInterval(bits))).toBe(openUnitInterval(bits));
    }
  });
});

describe("normalPair", () => {
  it("has the moments of a standard normal", () => {
    const sample: number[] = [];
    for (let i = 0; i < 200000; i++) {
      const pair = normalPair([i, 0], [4242, 0]);
      sample.push(pair[0], pair[1]);
    }
    const n = sample.length;
    const mean = sample.reduce((a, b) => a + b, 0) / n;
    const central = (power: number): number =>
      sample.reduce((a, b) => a + (b - mean) ** power, 0) / n;
    const variance = central(2);
    // Standard errors at n = 4e5 are 1/sqrt(n) = 1.6e-3 for the mean,
    // sqrt(2/n) = 2.2e-3 for the variance, sqrt(6/n) = 3.9e-3 for the skew and
    // sqrt(24/n) = 7.7e-3 for the kurtosis. Each bound below is about six of
    // its own standard errors, so this fails on a real defect and not on luck.
    expect(Math.abs(mean)).toBeLessThan(0.01);
    expect(Math.abs(variance - 1)).toBeLessThan(0.014);
    expect(Math.abs(central(3) / variance ** 1.5)).toBeLessThan(0.024);
    expect(Math.abs(central(4) / variance ** 2 - 3)).toBeLessThan(0.047);
  });

  it("puts the two halves of a pair in different directions", () => {
    // cos and sin of the same angle are uncorrelated over a uniform angle, but
    // only if the radius is shared and the angle is not. Getting this wrong --
    // by reusing one uniform for both -- gives a perfect correlation.
    let sum = 0;
    const n = 100000;
    for (let i = 0; i < n; i++) {
      const pair = normalPair([i, 11], [7, 0]);
      sum += (pair[0] as number) * (pair[1] as number);
    }
    expect(Math.abs(sum / n)).toBeLessThan(0.01);
  });
});

describe("thermostatNormal", () => {
  it("is a pure function of its indices", () => {
    expect(thermostatNormal(1, 2, 3, 0)).toBe(thermostatNormal(1, 2, 3, 0));
    expect(thermostatNormal(1, 2, 3, 2)).toBe(thermostatNormal(1, 2, 3, 2));
  });

  it("gives a particle three distinct deviates", () => {
    const values = [0, 1, 2].map((axis) => thermostatNormal(5, 0, 17, axis));
    expect(new Set(values).size).toBe(3);
  });

  it("decorrelates across axes, particles and steps", () => {
    // Three ways a counter layout goes wrong: axes sharing a draw, neighbouring
    // particles sharing one, or a step reusing the previous step's.
    const n = 40000;
    const pairs: readonly (readonly [string, (i: number) => [number, number]])[] = [
      ["axes 0 and 1", (i) => [thermostatNormal(3, 0, i, 0), thermostatNormal(3, 0, i, 1)]],
      ["axes 1 and 2", (i) => [thermostatNormal(3, 0, i, 1), thermostatNormal(3, 0, i, 2)]],
      ["neighbours", (i) => [thermostatNormal(3, 0, i, 0), thermostatNormal(3, 0, i + 1, 0)]],
      ["steps", (i) => [thermostatNormal(3, i, 9, 1), thermostatNormal(3, i + 1, 9, 1)]],
      ["seeds", (i) => [thermostatNormal(1, 0, i, 2), thermostatNormal(2, 0, i, 2)]],
    ];
    for (const [what, draw] of pairs) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const [a, b] = draw(i);
        sum += a * b;
      }
      // 1/sqrt(n) = 5e-3; the bound is four of those.
      expect(Math.abs(sum / n), what).toBeLessThan(0.02);
    }
  });

  it("binds a seed and a step into an indexed source", () => {
    const source = thermostatNormals(77, 4);
    expect(source(12, 1)).toBe(thermostatNormal(77, 4, 12, 1));
  });
});
