/**
 * L-BFGS against functions whose minima are known in closed form.
 *
 * The point of testing a minimiser on Rosenbrock rather than on a force field
 * is that the answer is not in question. If this reaches (1, 1, ..., 1) from a
 * standing start then the two-loop recursion, the ring buffer and the line
 * search are right, and anything that goes wrong later is the objective's
 * fault -- which is where the physics is, and where it is much harder to tell a
 * wrong gradient from a hard problem.
 */

import { describe, expect, it } from "vitest";
import { lbfgs, type Objective } from "../src/reference/lbfgs.js";

/** A quadratic with its minimum at `centre`, and a condition number. */
const scaledQuadratic =
  (centre: readonly number[], scales: readonly number[]): Objective =>
  (x, gradient) => {
    let value = 0;
    for (let i = 0; i < x.length; i++) {
      const d = (x[i] as number) - (centre[i] as number);
      value += (scales[i] as number) * d * d;
      gradient[i] = 2 * (scales[i] as number) * d;
    }
    return value;
  };

/** The extended Rosenbrock function; minimum 0 at every component equal to 1. */
const rosenbrock: Objective = (x, gradient) => {
  let value = 0;
  gradient.fill(0);
  for (let i = 0; i + 1 < x.length; i += 2) {
    const a = x[i] as number;
    const b = x[i + 1] as number;
    const t1 = 1 - a;
    const t2 = 10 * (b - a * a);
    value += t1 * t1 + t2 * t2;
    gradient[i] = (gradient[i] as number) - 2 * t1 - 400 * a * (b - a * a);
    gradient[i + 1] = (gradient[i + 1] as number) + 200 * (b - a * a);
  }
  return value;
};

describe("lbfgs", () => {
  it("finds a quadratic's minimum", async () => {
    const x = Float64Array.from([5, -3, 11]);
    const result = await lbfgs(x, scaledQuadratic([1, 2, 3], [1, 1, 1]), {
      epsilon: 1e-10,
    });
    expect(result.status).toBe("converged");
    expect(Array.from(x)).toEqual([
      expect.closeTo(1, 8),
      expect.closeTo(2, 8),
      expect.closeTo(3, 8),
    ]);
  });

  it("handles a badly conditioned quadratic", async () => {
    // Stiff bonds against soft dihedrals is the regime a force field lives in,
    // so the interesting question is how the cost grows with conditioning.
    // Measured on a 20-dimensional diagonal quadratic at epsilon 1e-9:
    //
    //   condition      1    1e2    1e4    1e6
    //   m = 6          2     82    604   4629   iterations
    //   m = 20         2     61    238    842
    //
    // Roughly the square root of the condition number at m = 6, and much
    // better once the memory can hold every distinct curvature. Steepest
    // descent would scale with the condition number itself, so this is the
    // shape that says the two-loop recursion is doing its job.
    const n = 20;
    const centre = Array.from({ length: n }, (_, i) => i * 0.1);
    const scales = Array.from({ length: n }, (_, i) => 10 ** (4 * (i / (n - 1)) - 2));
    const x = new Float64Array(n);
    const result = await lbfgs(x, scaledQuadratic(centre, scales), {
      epsilon: 1e-9,
      maxIterations: 2000,
    });
    expect(result.status).toBe("converged");
    for (let i = 0; i < n; i++) {
      expect(x[i]).toBeCloseTo(centre[i] as number, 4);
    }
  });

  it("minimises Rosenbrock in 2 dimensions", async () => {
    const x = Float64Array.from([-1.2, 1]);
    const result = await lbfgs(x, rosenbrock, { epsilon: 1e-10, maxIterations: 500 });
    expect(result.value).toBeLessThan(1e-12);
    expect(x[0]).toBeCloseTo(1, 6);
    expect(x[1]).toBeCloseTo(1, 6);
  });

  it("minimises Rosenbrock in 100 dimensions", async () => {
    const n = 100;
    const x = new Float64Array(n);
    for (let i = 0; i < n; i += 2) {
      x[i] = -1.2;
      x[i + 1] = 1;
    }
    const result = await lbfgs(x, rosenbrock, { epsilon: 1e-10, maxIterations: 2000 });
    expect(result.value).toBeLessThan(1e-10);
    for (let i = 0; i < n; i++) expect(x[i]).toBeCloseTo(1, 5);
  });

  it("stops at maxIterations and says so", async () => {
    const x = Float64Array.from([-1.2, 1]);
    const result = await lbfgs(x, rosenbrock, { epsilon: 1e-14, maxIterations: 3 });
    expect(result.status).toBe("max-iterations");
    expect(result.iterations).toBe(3);
  });

  it("reports converged without a single step when it starts at a minimum", async () => {
    const x = Float64Array.from([1, 2, 3]);
    const result = await lbfgs(x, scaledQuadratic([1, 2, 3], [1, 1, 1]));
    expect(result.status).toBe("converged");
    expect(result.iterations).toBe(0);
    // One evaluation, to find out. A minimiser that steps anyway would move a
    // configuration that was already where it belonged.
    expect(result.evaluations).toBe(1);
  });

  it("never returns a point worse than the one it started from", async () => {
    // The line search can run out of trials. When it does, the last accepted
    // point has to be restored -- returning the failed trial instead would make
    // a stalled minimisation actively harmful, and `x` is the caller's array.
    const x = Float64Array.from([-1.2, 1]);
    const start = new Float64Array(2);
    const initial = rosenbrock(x, start) as number;
    const result = await lbfgs(x, rosenbrock, {
      epsilon: 1e-16,
      maxLineSearch: 2,
      maxIterations: 50,
    });
    const check = new Float64Array(2);
    expect(rosenbrock(x, check)).toBeLessThanOrEqual(initial);
    expect(result.value).toBeLessThanOrEqual(initial);
  });

  it("uses its memory: more corrections converge in fewer iterations", async () => {
    const run = async (memory: number): Promise<number> => {
      const x = new Float64Array(40);
      for (let i = 0; i < 40; i += 2) {
        x[i] = -1.2;
        x[i + 1] = 1;
      }
      const result = await lbfgs(x, rosenbrock, {
        memory,
        epsilon: 1e-8,
        maxIterations: 5000,
      });
      return result.iterations;
    };
    // Not a performance assertion: a two-loop recursion that ignored its
    // history would converge like steepest descent and this would be flat.
    expect(await run(10)).toBeLessThan(await run(1));
  });

  it("gives the gradient ratio it stopped on", async () => {
    const x = Float64Array.from([5, -3, 11]);
    const result = await lbfgs(x, scaledQuadratic([1, 2, 3], [1, 1, 1]), {
      epsilon: 1e-8,
    });
    expect(result.gradientRatio).toBeLessThanOrEqual(1e-8);
  });
});
