import { describe, expect, it } from "vitest";
import {
  displacementNm,
  integrate,
  integrateCompensated,
  integrateCompensatedNm,
  integrateNm,
  positionFromWords,
} from "../src/reference/position.js";
import { toFixed } from "../src/reference/fixed-point.js";
import {
  dsAccumulate,
  dsAdd,
  dsAddF32,
  dsFromF32,
  dsToF64,
  f32Accumulate,
  f64Accumulate,
  twoSum,
} from "../src/reference/double-single.js";

/** The fixed-point format's resolution, in nm. */
const RESOLUTION_NM = 2 ** -32;

describe("positions in a 6.5 nm box", () => {
  const START = 6.5;

  it("f32 stops moving at a displacement below half an ulp", () => {
    // The 3HTB box is 6.52 nm, and the notebook's 1e-6 constraint tolerance on
    // a 0.1 nm X-H bond is 1e-7 nm -- below this coordinate's 4.77e-7 nm ulp.
    const displacements = new Float32Array(200_000).fill(Math.fround(1e-7));
    expect(f32Accumulate(START, displacements)).toBe(START);

    const exact = f64Accumulate(START, displacements);

    // Plain accumulation tracks it, but drifts: one displacement repeated has
    // the same conversion error every step, so the error adds up rather than
    // cancelling. Bounded by the worst case, half a resolution unit per step.
    const plainError = Math.abs(integrateNm(START, displacements) - exact);
    expect(plainError).toBeLessThanOrEqual((displacements.length * RESOLUTION_NM) / 2);
    expect(plainError).toBeGreaterThan(1e-6); // it really does drift

    // The compensated integrator carries what did not fit into the next step,
    // which bounds the outstanding error by one resolution unit forever.
    const compensated = integrateCompensated(START, displacements);
    const compensatedError = Math.abs(
      integrateCompensatedNm(START, displacements) - exact,
    );
    expect(compensatedError).toBeLessThan(RESOLUTION_NM * 2);
    expect(compensatedError).toBeLessThan(plainError / 1000);
    expect(Math.abs(compensated.residualNm)).toBeLessThanOrEqual(RESOLUTION_NM);

    expect(integrateCompensatedNm(START, displacements)).toBeGreaterThan(START + 0.019);
  });

  it("accumulates a realistic displacement history exactly", () => {
    let seed = 11;
    const random = (): number => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const displacements = new Float32Array(200_000);
    for (let i = 0; i < displacements.length; i++) {
      displacements[i] = Math.fround((random() * 2 - 1) * 5e-4);
    }
    // Integer addition of exactly-converted displacements: the only error is
    // the per-displacement conversion, bounded by the format's resolution.
    const want = toFixed(Math.fround(START));
    let sum = want;
    for (const d of displacements) sum += toFixed(d);
    expect(integrate(START, displacements)).toBe(sum);

    const f64 = f64Accumulate(START, displacements);
    expect(Math.abs(integrateNm(START, displacements) - f64)).toBeLessThan(
      displacements.length * RESOLUTION_NM,
    );
  });

  it("is order-independent, which f32 accumulation is not", () => {
    let seed = 29;
    const random = (): number => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const displacements = Array.from({ length: 10_000 }, () =>
      Math.fround((random() * 2 - 1) * 1e-3),
    );
    const forward = integrate(START, displacements);
    const backward = integrate(START, [...displacements].reverse());
    expect(backward).toBe(forward);

    const f32Forward = f32Accumulate(START, displacements);
    const f32Backward = f32Accumulate(START, [...displacements].reverse());
    // Not a requirement on f32, just the demonstration of why we do not use it.
    expect(f32Forward === f32Backward).toBe(false);
  });

  it("takes a displacement in fixed point and only then converts", () => {
    // How the engine reads positions: exact difference, then f32. A pair vector
    // is at most the cutoff, where f32 has plenty of room.
    const a = integrate(6.5, [Math.fround(1e-7)]);
    const b = integrate(6.5, []);
    // Recovered to the format's resolution, which is 2.3e-10 nm -- three
    // orders of magnitude below the 1e-7 nm the constraint tolerance needs.
    expect(Math.abs(displacementNm(a, b) - 1e-7)).toBeLessThanOrEqual(RESOLUTION_NM);
  });

  it("reassembles the two words the GPU writes", () => {
    const fixed = integrate(2.5, []);
    const lo = Number(fixed & 0xffffffffn);
    const hi = Number(fixed >> 32n);
    expect(positionFromWords(lo, hi)).toBe(fixed);
  });
});

/**
 * Dekker arithmetic is kept only as the reference for the device diagnostic.
 * These tests pin what it *should* compute, which is what makes the Chrome/Metal
 * measurement in docs/PRECISION.md a finding rather than a guess.
 */
describe("Dekker error-free arithmetic (diagnostic reference only)", () => {
  it("captures exactly the bits the sum dropped", () => {
    const [sum, err] = twoSum(Math.fround(2 ** 24), 1);
    expect(sum).toBe(2 ** 24);
    expect(err).toBe(1);
    expect(sum + err).toBe(2 ** 24 + 1);
  });

  it("would hold a value f32 cannot, if a backend computed it", () => {
    expect(dsToF64(dsAdd(dsFromF32(Math.fround(2 ** 24)), dsFromF32(1)))).toBe(
      2 ** 24 + 1,
    );
    expect(dsToF64(dsAddF32(dsFromF32(6.5), Math.fround(1e-7)))).toBeGreaterThan(6.5);
  });

  it("would not stagnate on the 1e-7 nm displacement", () => {
    const displacements = new Float32Array(200_000).fill(Math.fround(1e-7));
    const exact = f64Accumulate(6.5, displacements);
    expect(Math.abs(dsToF64(dsAccumulate(6.5, displacements)) - exact)).toBeLessThan(1e-9);
  });
});
