import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseState } from "../src/engine/state.js";
import { parseSystem, type NonbondedForce } from "../src/engine/system.js";
import { ewaldAlpha, pmeMinimumGridSize, smoothFftSize } from "../src/engine/units.js";
import {
  bsplineModuli,
  bsplineWeights,
  bsplineWeightsAndDerivatives,
} from "../src/reference/forces/bspline.js";
import { complexGrid, forward3d, inverse3d } from "../src/reference/dft.js";
import { PME_ORDER, reciprocalSpace } from "../src/reference/forces/pme.js";
import { netChargeCorrection, selfEnergy } from "../src/reference/forces/nonbonded.js";
import { fixture, FIXTURE_NAMES, referenceEnergy } from "./fixtures.js";

describe("B-splines", () => {
  it("conserves charge: the weights sum to one at any offset", () => {
    for (let w = 0; w < 1; w += 0.01) {
      const weights = bsplineWeights(w, PME_ORDER);
      let sum = 0;
      for (const value of weights) sum += value;
      expect(Math.abs(sum - 1)).toBeLessThan(1e-14);
    }
  });

  it("tracks the offset: the first moment moves exactly with w", () => {
    // The mean weight index is w + (order-2)/2, not w. The spline's support in
    // grid index runs from floor(u)-(order-1) to floor(u), so the assignment is
    // offset from the atom by a fixed amount -- and that offset is a global
    // shift of the whole mesh, which becomes a phase in Q(m) and cancels in
    // |Q(m)|^2. What has to hold is that the moment follows w with slope one.
    const offset = (PME_ORDER - 2) / 2;
    for (let w = 0; w < 1; w += 0.01) {
      const weights = bsplineWeights(w, PME_ORDER);
      let moment = 0;
      for (let k = 0; k < PME_ORDER; k++) moment += (weights[k] as number) * k;
      expect(Math.abs(moment - (w + offset))).toBeLessThan(1e-13);
    }
  });

  it("is non-negative everywhere", () => {
    for (let w = 0; w < 1; w += 0.01) {
      for (const value of bsplineWeights(w, PME_ORDER)) {
        expect(value).toBeGreaterThanOrEqual(0);
      }
    }
  });

  it("has derivatives that match a central difference", () => {
    const h = 1e-6;
    for (let w = 0.1; w < 0.9; w += 0.05) {
      const { derivatives } = bsplineWeightsAndDerivatives(w, PME_ORDER);
      const plus = bsplineWeights(w + h, PME_ORDER);
      const minus = bsplineWeights(w - h, PME_ORDER);
      for (let k = 0; k < PME_ORDER; k++) {
        const numerical = ((plus[k] as number) - (minus[k] as number)) / (2 * h);
        expect(Math.abs((derivatives[k] as number) - numerical)).toBeLessThan(1e-7);
      }
    }
  });

  it("gives positive moduli for every frequency", () => {
    for (const size of [25, 32, 54, 64]) {
      for (const value of bsplineModuli(size, PME_ORDER)) {
        expect(value).toBeGreaterThan(0);
      }
    }
  });
});

describe("the 3D transform", () => {
  it("round-trips to n times the input", () => {
    const size: [number, number, number] = [8, 6, 10];
    const grid = complexGrid(size);
    let seed = 5;
    const random = (): number => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const original = Float64Array.from(grid.data, () => random() * 2 - 1);
    grid.data.set(original);
    forward3d(grid);
    inverse3d(grid);
    const n = size[0] * size[1] * size[2];
    for (let i = 0; i < grid.data.length; i++) {
      expect(Math.abs((grid.data[i] as number) / n - (original[i] as number))).toBeLessThan(1e-12);
    }
  });

  it("transforms a delta to a constant", () => {
    const grid = complexGrid([4, 4, 4]);
    grid.data[0] = 1;
    forward3d(grid);
    for (let i = 0; i < 64; i++) {
      expect(grid.data[i * 2]).toBeCloseTo(1, 12);
      expect(grid.data[i * 2 + 1]).toBeCloseTo(0, 12);
    }
  });
});

interface NonbondedFacts {
  readonly PME: { readonly alpha_per_nm: number; readonly grid: number[]; readonly reciprocal_group: number };
  readonly decomposition: Record<string, number>;
}

const small = (() => {
  try {
    return {
      system: parseSystem(readFileSync("test/fixtures/nonbonded-system.xml", "utf8")),
      state: parseState(readFileSync("test/fixtures/nonbonded-state.xml", "utf8")),
      facts: JSON.parse(
        readFileSync("test/fixtures/nonbonded-facts.json", "utf8"),
      ) as NonbondedFacts,
    };
  } catch {
    return null;
  }
})();

describe("reciprocal space against OpenMM", () => {
  it("reproduces the reciprocal group of the committed fixture", () => {
    expect(small).not.toBeNull();
    const force = small!.system.forces.find(
      (f) => f.type === "NonbondedForce",
    ) as NonbondedForce;
    const alpha = small!.facts.PME.alpha_per_nm;
    const grid = small!.facts.PME.grid as unknown as [number, number, number];
    const { energy } = reciprocalSpace(
      force,
      small!.state.positions,
      small!.system.boxVectors,
      alpha,
      grid,
    );
    // OpenMM's reciprocal group is the mesh sum, the self energy and the
    // net-charge correction. This fixture carries a net charge of -0.416 e on
    // purpose: with the correction left out the comparison is off by 0.32
    // kJ/mol, which is how the term was found.
    const box = small!.system.boxVectors[0][0] as number;
    const volume = box ** 3;
    const total =
      energy + selfEnergy(force, alpha) + netChargeCorrection(force, alpha, volume);
    expect(Math.abs(total - small!.facts.PME.reciprocal_group)).toBeLessThan(1e-4);
  });

  it("uses the same grid OpenMM chose", () => {
    const force = small!.system.forces.find(
      (f) => f.type === "NonbondedForce",
    ) as NonbondedForce;
    const alpha = ewaldAlpha(force.cutoff, force.ewaldTolerance);
    const box = small!.system.boxVectors[0][0];
    const minimum = pmeMinimumGridSize(alpha, box, force.ewaldTolerance);
    // OpenMM's Reference platform uses the minimum as-is; the engine rounds up
    // to a transform-friendly size, which must not be smaller.
    expect(minimum).toBeLessThanOrEqual(small!.facts.PME.grid[0] as number);
    expect(smoothFftSize(minimum)).toBeGreaterThanOrEqual(minimum);
  });

  it("gives forces that match a numerical derivative of its own energy", () => {
    const force = small!.system.forces.find(
      (f) => f.type === "NonbondedForce",
    ) as NonbondedForce;
    const alpha = small!.facts.PME.alpha_per_nm;
    const grid = small!.facts.PME.grid as unknown as [number, number, number];
    const analytic = new Float64Array(small!.system.particleCount * 3);
    reciprocalSpace(
      force,
      small!.state.positions,
      small!.system.boxVectors,
      alpha,
      grid,
      analytic,
    );
    const positions = Float64Array.from(small!.state.positions);
    const h = 1e-5;
    const energyAt = (): number =>
      reciprocalSpace(force, positions, small!.system.boxVectors, alpha, grid).energy;
    for (let i = 0; i < positions.length; i++) {
      const original = positions[i] as number;
      positions[i] = original + h;
      const plus = energyAt();
      positions[i] = original - h;
      const minus = energyAt();
      positions[i] = original;
      const numerical = -(plus - minus) / (2 * h);
      const magnitude = Math.max(Math.abs(numerical), Math.abs(analytic[i] as number), 1);
      expect(Math.abs((analytic[i] as number) - numerical) / magnitude).toBeLessThan(1e-5);
    }
  });
});

for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  describe.skipIf(f === null)(`reciprocal space for ${name}`, () => {
    it("reproduces OpenMM's reciprocal force group", () => {
      const force = f!.system.forces.find(
        (x) => x.type === "NonbondedForce",
      ) as NonbondedForce;
      const alpha = f!.facts.pme.alpha_per_nm;
      // OpenMM's Reference platform uses the exact minimum grid, so that is
      // what this comparison has to use -- not the smooth size the engine
      // rounds up to, which would be a different approximation.
      const grid = f!.facts.pme.grid_reference_platform as unknown as [number, number, number];
      const { energy } = reciprocalSpace(
        force,
        f!.state.positions,
        f!.system.boxVectors,
        alpha,
        grid,
      );
      const box = f!.system.boxVectors;
      const volume =
        (box[0][0] as number) * (box[1][1] as number) * (box[2][2] as number);
      const total =
        energy + selfEnergy(force, alpha) + netChargeCorrection(force, alpha, volume);
      const want = referenceEnergy(f!, "NonbondedForce[reciprocal]");
      expect(Math.abs(total - want) / Math.abs(want)).toBeLessThan(1e-9);
    });
  });
}
