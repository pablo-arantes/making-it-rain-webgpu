/**
 * Cardinal B-splines, the interpolation smooth PME is built on.
 *
 * `M_p` is supported on [0, p] and is what spreads a point charge onto the
 * mesh.  Order 5 is what the notebook's OpenMM uses, and what the engine uses.
 *
 * Two properties make these testable without reference to PME at all, and
 * test/pme-reference.test.ts checks both: the weights sum to one for any
 * offset (so spreading conserves charge), and the first moment reproduces the
 * offset (so the charge lands in the right place).
 */

/**
 * The `order` B-spline weights for a point at fractional offset `w` in [0, 1).
 *
 * Returns weights for the grid points at `floor(u) - order + 1 + k` for
 * k = 0..order-1, which is the indexing the spreading and interpolation both
 * use.  Computed by the standard recursion from the order-2 hat function.
 */
export function bsplineWeights(w: number, order: number): Float64Array {
  const theta = new Float64Array(order);
  theta[order - 1] = 0;
  theta[1] = w;
  theta[0] = 1 - w;
  for (let k = 3; k <= order - 1; k++) {
    const denominator = 1 / (k - 1);
    theta[k - 1] = denominator * w * (theta[k - 2] as number);
    for (let j = 1; j < k - 1; j++) {
      theta[k - j - 1] =
        denominator *
        ((w + j) * (theta[k - j - 2] as number) +
          (k - j - w) * (theta[k - j - 1] as number));
    }
    theta[0] = denominator * (1 - w) * (theta[0] as number);
  }
  // The last level, which also gives the derivative below.
  const denominator = 1 / (order - 1);
  theta[order - 1] = denominator * w * (theta[order - 2] as number);
  for (let j = 1; j < order - 1; j++) {
    theta[order - j - 1] =
      denominator *
      ((w + j) * (theta[order - j - 2] as number) +
        (order - j - w) * (theta[order - j - 1] as number));
  }
  theta[0] = denominator * (1 - w) * (theta[0] as number);
  return theta;
}

/**
 * The weights and their derivatives with respect to `w`.
 *
 * dM_p/du = M_{p-1}(u) - M_{p-1}(u - 1), so the derivative falls out of the
 * order-1 weights computed one step earlier in the same recursion.
 */
export function bsplineWeightsAndDerivatives(
  w: number,
  order: number,
): { weights: Float64Array; derivatives: Float64Array } {
  // One level short of the full order.
  const lower = new Float64Array(order);
  lower[1] = w;
  lower[0] = 1 - w;
  for (let k = 3; k <= order - 1; k++) {
    const denominator = 1 / (k - 1);
    lower[k - 1] = denominator * w * (lower[k - 2] as number);
    for (let j = 1; j < k - 1; j++) {
      lower[k - j - 1] =
        denominator *
        ((w + j) * (lower[k - j - 2] as number) +
          (k - j - w) * (lower[k - j - 1] as number));
    }
    lower[0] = denominator * (1 - w) * (lower[0] as number);
  }
  const derivatives = new Float64Array(order);
  derivatives[0] = -(lower[0] as number);
  for (let j = 1; j < order; j++) {
    derivatives[j] = (lower[j - 1] as number) - (lower[j] as number);
  }
  // Then the final recursion level for the weights themselves.
  const weights = Float64Array.from(lower);
  const denominator = 1 / (order - 1);
  weights[order - 1] = denominator * w * (weights[order - 2] as number);
  for (let j = 1; j < order - 1; j++) {
    weights[order - j - 1] =
      denominator *
      ((w + j) * (weights[order - j - 2] as number) +
        (order - j - w) * (weights[order - j - 1] as number));
  }
  weights[0] = denominator * (1 - w) * (weights[0] as number);
  return { weights, derivatives };
}

/**
 * The B-spline modulus correction, one array per axis.
 *
 * Spreading onto a mesh and transforming is not the same as transforming point
 * charges: the mesh assignment convolves each charge with the spline, so the
 * reciprocal sum has to be divided by the spline's own structure factor.  This
 * returns 1/|b(m)|^2 for every frequency index, which is the factor the
 * convolution multiplies by.
 */
export function bsplineModuli(gridSize: number, order: number): Float64Array {
  // The spline evaluated at its integer knots, which is its own structure
  // factor's coefficient sequence.
  const data = new Float64Array(gridSize);
  const atKnots = bsplineWeights(0, order);
  for (let i = 0; i < order; i++) {
    if (i < gridSize) data[i] = atKnots[i] as number;
  }
  const moduli = new Float64Array(gridSize);
  for (let m = 0; m < gridSize; m++) {
    let real = 0;
    let imaginary = 0;
    for (let k = 0; k < gridSize; k++) {
      const angle = (2 * Math.PI * m * k) / gridSize;
      real += (data[k] as number) * Math.cos(angle);
      imaginary += (data[k] as number) * Math.sin(angle);
    }
    moduli[m] = real * real + imaginary * imaginary;
  }
  // A zero would mean this frequency carries no information the spline can
  // represent. For an even order on a grid at least as large as the order it
  // does not happen; averaging the neighbours is the standard guard.
  for (let m = 0; m < gridSize; m++) {
    if ((moduli[m] as number) < 1e-7) {
      const previous = moduli[(m - 1 + gridSize) % gridSize] as number;
      const next = moduli[(m + 1) % gridSize] as number;
      moduli[m] = 0.5 * (previous + next);
    }
  }
  return moduli;
}
