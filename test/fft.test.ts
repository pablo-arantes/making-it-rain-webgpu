import { describe, expect, it } from "vitest";
import { complexGrid, forward3d, inverse3d } from "../src/reference/dft.js";
import { fft3d, NotSmooth, planFft, RADICES } from "../src/reference/fft.js";
import { smoothFftSize } from "../src/engine/units.js";

const random = (seed: number) => {
  let state = seed;
  return (): number =>
    (state = (state * 1103515245 + 12345) % 2147483648) / 2147483648;
};

describe("planning", () => {
  it("factors with the largest radix first", () => {
    expect(planFft(54).stages.map((s) => s.radix)).toEqual([3, 3, 3, 2]);
    expect(planFft(25).stages.map((s) => s.radix)).toEqual([5, 5]);
    expect(planFft(64).stages.map((s) => s.radix)).toEqual([2, 2, 2, 2, 2, 2]);
    expect(planFft(7).stages.map((s) => s.radix)).toEqual([7]);
  });

  it("tracks Ns as the product of the earlier radices", () => {
    const plan = planFft(54);
    let ns = 1;
    for (const stage of plan.stages) {
      expect(stage.ns).toBe(ns);
      ns *= stage.radix;
    }
    expect(ns).toBe(54);
  });

  it("handles a length of one", () => {
    expect(planFft(1).stages).toEqual([]);
  });

  it("refuses a length with a factor it has no radix for", () => {
    expect(() => planFft(11)).toThrow(NotSmooth);
    expect(() => planFft(26)).toThrow(/not 2\/3\/5\/7-smooth/);
  });

  it("can plan every size smoothFftSize produces", () => {
    for (let n = 1; n <= 300; n++) {
      const size = smoothFftSize(n);
      expect(() => planFft(size)).not.toThrow();
      let product = 1;
      for (const stage of planFft(size).stages) {
        expect(RADICES).toContain(stage.radix as 2 | 3 | 5 | 7);
        product *= stage.radix;
      }
      expect(product).toBe(size);
    }
  });
});

describe("the mixed-radix transform against the direct one", () => {
  // Shapes chosen to exercise every radix, mixed radices, and non-cubic grids.
  const shapes: ReadonlyArray<readonly [number, number, number]> = [
    [2, 2, 2],
    [3, 3, 3],
    [5, 5, 5],
    [7, 7, 7],
    [4, 6, 10],
    [25, 25, 25],
    [54, 2, 3],
    [8, 9, 5],
  ];

  it.each(shapes)("agrees forward on %ix%ix%i", (nx, ny, nz) => {
    const next = random(17 + nx * 100 + ny * 10 + nz);
    const total = nx * ny * nz;
    const original = Float64Array.from({ length: total * 2 }, () => next() * 2 - 1);

    const direct = complexGrid([nx, ny, nz]);
    direct.data.set(original);
    forward3d(direct);

    const fast = Float64Array.from(original);
    fft3d(fast, [nx, ny, nz], -1);

    let worst = 0;
    let scale = 0;
    for (let i = 0; i < fast.length; i++) {
      worst = Math.max(worst, Math.abs((fast[i] as number) - (direct.data[i] as number)));
      scale = Math.max(scale, Math.abs(direct.data[i] as number));
    }
    expect(worst / scale).toBeLessThan(1e-13);
  });

  it.each(shapes)("agrees inverse on %ix%ix%i", (nx, ny, nz) => {
    const next = random(29 + nx);
    const total = nx * ny * nz;
    const original = Float64Array.from({ length: total * 2 }, () => next() * 2 - 1);

    const direct = complexGrid([nx, ny, nz]);
    direct.data.set(original);
    inverse3d(direct);

    const fast = Float64Array.from(original);
    fft3d(fast, [nx, ny, nz], 1);

    let worst = 0;
    let scale = 0;
    for (let i = 0; i < fast.length; i++) {
      worst = Math.max(worst, Math.abs((fast[i] as number) - (direct.data[i] as number)));
      scale = Math.max(scale, Math.abs(direct.data[i] as number));
    }
    expect(worst / scale).toBeLessThan(1e-13);
  });
});

describe("transform properties", () => {
  it("round-trips to n times the input", () => {
    const size: [number, number, number] = [54, 25, 8];
    const total = size[0] * size[1] * size[2];
    const next = random(41);
    const original = Float64Array.from({ length: total * 2 }, () => next() * 2 - 1);
    const data = Float64Array.from(original);
    fft3d(data, size, -1);
    fft3d(data, size, 1);
    for (let i = 0; i < data.length; i++) {
      expect(Math.abs((data[i] as number) / total - (original[i] as number))).toBeLessThan(
        1e-12,
      );
    }
  });

  it("transforms a delta to a constant", () => {
    const size: [number, number, number] = [25, 9, 4];
    const data = new Float64Array(size[0] * size[1] * size[2] * 2);
    data[0] = 1;
    fft3d(data, size, -1);
    for (let i = 0; i < size[0] * size[1] * size[2]; i++) {
      expect(data[i * 2]).toBeCloseTo(1, 12);
      expect(data[i * 2 + 1]).toBeCloseTo(0, 12);
    }
  });

  it("is linear", () => {
    const size: [number, number, number] = [9, 5, 2];
    const total = size[0] * size[1] * size[2];
    const next = random(53);
    const a = Float64Array.from({ length: total * 2 }, () => next() * 2 - 1);
    const b = Float64Array.from({ length: total * 2 }, () => next() * 2 - 1);
    const sum = Float64Array.from(a, (value, i) => value + (b[i] as number));
    fft3d(a, size, -1);
    fft3d(b, size, -1);
    fft3d(sum, size, -1);
    for (let i = 0; i < sum.length; i++) {
      expect(Math.abs((sum[i] as number) - ((a[i] as number) + (b[i] as number)))).toBeLessThan(
        1e-12,
      );
    }
  });
});
