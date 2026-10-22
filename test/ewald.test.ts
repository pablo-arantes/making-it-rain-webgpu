/**
 * The exact Ewald sum against the mesh PME, on a rectangular box.
 *
 * 🔴 This validates the *oracle*, which has to happen before the oracle is used
 * to validate anything. `ewaldReciprocal` shares nothing with `reciprocalSpace`
 * but the physics — no B-spline, no grid, no transform, no index arithmetic —
 * so agreement on a box where both are known-good is what establishes that the
 * constant, the sign and the lattice convention are right. Only then is it
 * worth anything as the reference for a triclinic cell, where `reciprocalSpace`
 * is the thing under test.
 *
 * Agreement is to PME's own accuracy, not to round-off: the mesh path is an
 * approximation with a spline order and a grid, and the exact sum is not. The
 * tolerance here is therefore a statement about PME's error, which is why it is
 * asserted as a band rather than a bound — a suspiciously *small* difference
 * would mean the two are not independent.
 */

import { describe, expect, it } from "vitest";
import { ewaldHalfWidths, ewaldReciprocal } from "../src/reference/forces/ewald.js";
import { reciprocalSpace } from "../src/reference/forces/pme.js";
import {
  boxVectorsForShape,
  inverseBox,
  reciprocalVector,
  type BoxVectors,
} from "../src/reference/triclinic.js";
import type { NonbondedForce, Vec3 } from "../src/engine/system.js";

/** A small neutral charge set in a cube, which both paths can handle. */
function makeSystem(
  particleCount: number,
  width: number,
  seed = 7,
): {
  force: NonbondedForce;
  positions: Float64Array;
  box: BoxVectors;
} {
  // A tiny deterministic generator, so the test does not depend on a library.
  let state = seed;
  const next = (): number => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  const charge: number[] = [];
  const positions = new Float64Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    charge.push(i % 2 === 0 ? 0.6 : -0.6);
    for (let axis = 0; axis < 3; axis++) {
      positions[i * 3 + axis] = next() * width;
    }
  }
  const force: NonbondedForce = {
    type: "NonbondedForce",
    particleCount,
    charge,
    sigma: new Array(particleCount).fill(0.3),
    epsilon: new Array(particleCount).fill(0.5),
    cutoff: 0.9,
    ewaldTolerance: 5e-4,
    useDispersionCorrection: false,
    exceptions: [],
    exceptionCount: 0,
  } as unknown as NonbondedForce;
  return {
    force,
    positions,
    box: [[width, 0, 0], [0, width, 0], [0, 0, width]],
  };
}

describe("the exact Ewald sum", () => {
  it("collapses to 2*pi*m/L on a rectangular box", () => {
    const box: BoxVectors = [[2, 0, 0], [0, 3, 0], [0, 0, 5]];
    const inverse = inverseBox(box);
    const k = reciprocalVector(2, -3, 4, inverse);
    expect(k[0] as number).toBeCloseTo((2 * Math.PI * 2) / 2, 12);
    expect(k[1] as number).toBeCloseTo((2 * Math.PI * -3) / 3, 12);
    expect(k[2] as number).toBeCloseTo((2 * Math.PI * 4) / 5, 12);
  });

  it("puts a skewed cell's reciprocal vectors orthogonal to the right planes", () => {
    // g_i . a_j = 2 pi delta_ij is the definition, so check it directly rather
    // than trusting the algebra that produced it.
    const box = boxVectorsForShape("dodecahedron", 3);
    const inverse = inverseBox(box);
    for (let i = 0; i < 3; i++) {
      const g = reciprocalVector(i === 0 ? 1 : 0, i === 1 ? 1 : 0, i === 2 ? 1 : 0, inverse);
      for (let j = 0; j < 3; j++) {
        const a = box[j] as Vec3;
        const dot =
          (g[0] as number) * (a[0] as number) +
          (g[1] as number) * (a[1] as number) +
          (g[2] as number) * (a[2] as number);
        expect(dot).toBeCloseTo(i === j ? 2 * Math.PI : 0, 10);
      }
    }
  });

  it("takes *fewer* lattice indices for a narrower cell, not more", () => {
    // The direction is counterintuitive and worth pinning. A narrow cell has
    // *large* reciprocal vectors, so each step in `m` moves further in k and
    // fewer steps reach the point where exp(-k^2/4a^2) is negligible. The
    // reciprocal-space work therefore falls as the box shrinks, which is the
    // opposite of the real-space work.
    const wide = ewaldHalfWidths(boxVectorsForShape("cube", 4), 3.1);
    const narrow = ewaldHalfWidths(boxVectorsForShape("cube", 2), 3.1);
    expect(narrow[0] as number).toBeLessThan(wide[0] as number);
    // Halving the box roughly halves the count, because |k| per step doubles.
    // 19 and 10 rather than 19 and 9.5: the count is a ceiling.
    const ratio = (wide[0] as number) / (narrow[0] as number);
    expect(ratio).toBeGreaterThan(1.7);
    expect(ratio).toBeLessThan(2.2);
  });

  it("🔴 agrees with the mesh PME on a cube, to PME's own accuracy", () => {
    const { force, positions, box } = makeSystem(40, 2.4);
    const alpha = 3.0;
    const mesh = reciprocalSpace(force, positions, box, alpha, [32, 32, 32]);
    const exact = ewaldReciprocal(force.charge, positions, box, alpha);

    expect(exact.vectors).toBeGreaterThan(1000);
    const relative = Math.abs(mesh.energy - exact.energy) / Math.abs(exact.energy);
    // PME with an order-5 spline on a 32³ mesh is good to parts in 1e6 here.
    expect(relative).toBeLessThan(1e-4);
    // And not *identical*, which would mean they are not independent.
    expect(relative).toBeGreaterThan(0);
  });

  it("agrees on the forces too, which is the sensitive comparison", () => {
    const { force, positions, box } = makeSystem(30, 2.4);
    const alpha = 3.0;
    const meshForces = new Float64Array(positions.length);
    reciprocalSpace(force, positions, box, alpha, [32, 32, 32], meshForces);
    const exactForces = new Float64Array(positions.length);
    ewaldReciprocal(force.charge, positions, box, alpha, { forces: exactForces });

    let numerator = 0;
    let denominator = 0;
    for (let k = 0; k < positions.length; k++) {
      const difference = (meshForces[k] as number) - (exactForces[k] as number);
      numerator += difference * difference;
      denominator += (exactForces[k] as number) ** 2;
    }
    expect(denominator).toBeGreaterThan(0);
    expect(Math.sqrt(numerator / denominator)).toBeLessThan(1e-3);
  });
});
