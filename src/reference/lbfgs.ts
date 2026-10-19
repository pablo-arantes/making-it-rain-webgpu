/**
 * L-BFGS with a backtracking strong-Wolfe line search.
 *
 * OpenMM minimises with libLBFGS (Naoaki Okazaki's C port of Nocedal's
 * original), configured in `ReferenceMinimize.cpp` as
 * `LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE` with every other parameter left
 * at its default. This is that algorithm with those defaults, so the search
 * follows the same path to the same basin.
 *
 * Not bit-identical to libLBFGS, and it does not need to be. A minimiser is
 * judged by where it stops, not by the route: the fixed point is set by the
 * objective -- which *is* replicated exactly, including the constraint springs
 * and their schedule (src/reference/minimise.ts) -- while the line search only
 * decides how quickly it gets there. test/lbfgs.test.ts pins the implementation
 * against functions whose minima are known in closed form, and
 * test/minimise.test.ts pins the whole thing against OpenMM's own result.
 *
 * The evaluator is async because on the device it is a dispatch and a readback.
 * That makes this a little heavier than the textbook version and is the only
 * concession to where it runs.
 */

/** The objective: a value, and the gradient written into `gradient`. */
export type Objective = (
  x: Float64Array,
  gradient: Float64Array,
) => Promise<number> | number;

export interface LbfgsOptions {
  /** How many correction pairs to keep. libLBFGS's default is 6. */
  readonly memory?: number;
  /**
   * Convergence: `|g| / max(|x|, 1) <= epsilon`, both L2 norms.
   *
   * This is libLBFGS's test, and it is worth writing out what OpenMM's choice
   * of `epsilon` turns it into. OpenMM passes `tolerance / norm` where `norm`
   * is `sqrt(mean |x_i|^2)` clamped below at 1, and `|x| = sqrt(N) * norm`, so
   * the test reduces to `sqrt(sum |g_i|^2 / N) <= tolerance` -- the root mean
   * square force per particle. That is what OpenMM's documented tolerance of
   * 10 kJ/mol/nm actually means.
   */
  readonly epsilon?: number;
  /** Zero means no limit, as in libLBFGS. */
  readonly maxIterations?: number;
  readonly maxLineSearch?: number;
  /** Armijo's sufficient-decrease parameter. */
  readonly ftol?: number;
  /** The curvature parameter for the Wolfe conditions. */
  readonly wolfe?: number;
  readonly minStep?: number;
  readonly maxStep?: number;
}

export type LbfgsStatus =
  | "converged"
  | "max-iterations"
  | "line-search-failed"
  | "not-a-descent-direction"
  | "step-too-small"
  | "step-too-large";

export interface LbfgsResult {
  readonly value: number;
  readonly iterations: number;
  readonly evaluations: number;
  readonly status: LbfgsStatus;
  /** `|g| / max(|x|, 1)` at the end. */
  readonly gradientRatio: number;
}

const DEFAULTS = {
  memory: 6,
  epsilon: 1e-5,
  maxIterations: 0,
  maxLineSearch: 40,
  ftol: 1e-4,
  wolfe: 0.9,
  minStep: 1e-20,
  maxStep: 1e20,
} as const;

/** The step is halved on a failed Armijo test and grown on a failed curvature one. */
const DECREASE = 0.5;
const INCREASE = 2.1;

const dot = (a: Float64Array, b: Float64Array): number => {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] as number) * (b[i] as number);
  return sum;
};

const norm2 = (a: Float64Array): number => Math.sqrt(dot(a, a));

/**
 * Minimise `objective` from `x`, in place.
 *
 * Returns when the gradient test is met, when `maxIterations` is reached, or
 * when the line search cannot make progress -- which is not a failure of the
 * caller's and is reported rather than thrown, because a minimisation that
 * stalls near a minimum has still done its job.
 */
export async function lbfgs(
  x: Float64Array,
  objective: Objective,
  options: LbfgsOptions = {},
): Promise<LbfgsResult> {
  const memory = options.memory ?? DEFAULTS.memory;
  const epsilon = options.epsilon ?? DEFAULTS.epsilon;
  const maxIterations = options.maxIterations ?? DEFAULTS.maxIterations;
  const maxLineSearch = options.maxLineSearch ?? DEFAULTS.maxLineSearch;
  const ftol = options.ftol ?? DEFAULTS.ftol;
  const wolfe = options.wolfe ?? DEFAULTS.wolfe;
  const minStep = options.minStep ?? DEFAULTS.minStep;
  const maxStep = options.maxStep ?? DEFAULTS.maxStep;

  const n = x.length;
  const gradient = new Float64Array(n);
  const direction = new Float64Array(n);
  const previousX = new Float64Array(n);
  const previousGradient = new Float64Array(n);
  // The correction pairs, as a ring buffer.
  const s: Float64Array[] = [];
  const y: Float64Array[] = [];
  const ys: number[] = [];
  const alpha = new Float64Array(memory);
  for (let i = 0; i < memory; i++) {
    s.push(new Float64Array(n));
    y.push(new Float64Array(n));
    ys.push(0);
  }

  let evaluations = 0;
  let value = await objective(x, gradient);
  evaluations++;

  const ratio = (): number => norm2(gradient) / Math.max(norm2(x), 1);
  if (ratio() <= epsilon) {
    return {
      value,
      iterations: 0,
      evaluations,
      status: "converged",
      gradientRatio: ratio(),
    };
  }

  for (let i = 0; i < n; i++) direction[i] = -(gradient[i] as number);
  // The first step is scaled to the gradient's magnitude; later ones start at
  // one, because the inverse-Hessian approximation has already scaled it.
  let step = 1 / norm2(direction);
  let end = 0;
  let stored = 0;

  for (let iteration = 1; ; iteration++) {
    previousX.set(x);
    previousGradient.set(gradient);

    // --- Backtracking line search with the strong Wolfe conditions ---------
    const initialDirectional = dot(gradient, direction);
    if (initialDirectional > 0) {
      x.set(previousX);
      return {
        value,
        iterations: iteration - 1,
        evaluations,
        status: "not-a-descent-direction",
        gradientRatio: ratio(),
      };
    }
    const initialValue = value;
    const armijoSlope = ftol * initialDirectional;
    let searchStatus: LbfgsStatus | "ok" = "line-search-failed";
    for (let trial = 0; trial < maxLineSearch; trial++) {
      for (let i = 0; i < n; i++) {
        x[i] = (previousX[i] as number) + step * (direction[i] as number);
      }
      value = await objective(x, gradient);
      evaluations++;

      let width: number;
      if (value > initialValue + step * armijoSlope) {
        width = DECREASE;
      } else {
        const directional = dot(gradient, direction);
        if (directional < wolfe * initialDirectional) {
          width = INCREASE;
        } else if (directional > -wolfe * initialDirectional) {
          // The strong condition: the slope must be small in magnitude, not
          // merely less negative. Without it a step can overshoot the minimum
          // and still be accepted.
          width = DECREASE;
        } else {
          searchStatus = "ok";
          break;
        }
      }
      if (step < minStep) {
        searchStatus = "step-too-small";
        break;
      }
      if (step > maxStep) {
        searchStatus = "step-too-large";
        break;
      }
      step *= width;
    }
    if (searchStatus !== "ok") {
      // Nothing better was found, so the last accepted point stands.
      x.set(previousX);
      gradient.set(previousGradient);
      value = initialValue;
      return {
        value,
        iterations: iteration - 1,
        evaluations,
        status: searchStatus,
        gradientRatio: ratio(),
      };
    }

    const gradientRatio = ratio();
    if (gradientRatio <= epsilon) {
      return { value, iterations: iteration, evaluations, status: "converged", gradientRatio };
    }
    if (maxIterations !== 0 && iteration >= maxIterations) {
      return {
        value,
        iterations: iteration,
        evaluations,
        status: "max-iterations",
        gradientRatio,
      };
    }

    // --- Record the correction pair ---------------------------------------
    const sEnd = s[end] as Float64Array;
    const yEnd = y[end] as Float64Array;
    for (let i = 0; i < n; i++) {
      sEnd[i] = (x[i] as number) - (previousX[i] as number);
      yEnd[i] = (gradient[i] as number) - (previousGradient[i] as number);
    }
    const sy = dot(yEnd, sEnd);
    const yy = dot(yEnd, yEnd);
    ys[end] = sy;
    stored = Math.min(stored + 1, memory);
    end = (end + 1) % memory;

    // --- Two-loop recursion: direction = -H g -----------------------------
    for (let i = 0; i < n; i++) direction[i] = -(gradient[i] as number);
    let j = end;
    for (let count = 0; count < stored; count++) {
      j = (j + memory - 1) % memory;
      const sj = s[j] as Float64Array;
      const yj = y[j] as Float64Array;
      const a = dot(sj, direction) / (ys[j] as number);
      alpha[j] = a;
      for (let i = 0; i < n; i++) {
        direction[i] = (direction[i] as number) - a * (yj[i] as number);
      }
    }
    // Scaling by ys/yy, which is the only use the newest pair's `yy` has.
    const scale = sy / yy;
    for (let i = 0; i < n; i++) direction[i] = (direction[i] as number) * scale;
    for (let count = 0; count < stored; count++) {
      const sj = s[j] as Float64Array;
      const yj = y[j] as Float64Array;
      const beta = dot(yj, direction) / (ys[j] as number);
      const a = alpha[j] as number;
      for (let i = 0; i < n; i++) {
        direction[i] = (direction[i] as number) + (a - beta) * (sj[i] as number);
      }
      j = (j + 1) % memory;
    }
    step = 1;
  }
}
