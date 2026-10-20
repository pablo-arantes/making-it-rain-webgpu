/**
 * Eigenvalues and eigenvectors of a real symmetric matrix, by Jacobi rotations.
 *
 * Needed twice, for jobs that look unrelated. Best-fit superposition reduces to
 * the largest eigenvalue of a 4x4 symmetric matrix (the quaternion method), and
 * principal component analysis is the eigendecomposition of a coordinate
 * covariance matrix — 60x60 for a twenty-residue peptide, 489x489 for a protein
 * the size of T4 lysozyme.
 *
 * Jacobi rather than anything cleverer, for three reasons. It is unconditionally
 * stable for symmetric matrices and needs no balancing or shifting; it produces
 * a full orthogonal eigenvector set rather than one vector at a time, which is
 * what PCA wants; and it is short enough to read. It is O(n^3) with a largish
 * constant, which at 489x489 is a fraction of a second — and the alternative is
 * a linear algebra dependency for one routine.
 *
 * Returns eigenvalues in **descending** order with their eigenvectors, because
 * every caller here wants the largest first.
 */

export interface Eigen {
  /** Descending. */
  readonly values: Float64Array;
  /** Column `k` is the eigenvector for `values[k]`: `vectors[i * n + k]`. */
  readonly vectors: Float64Array;
}

/**
 * `matrix` is `n * n`, row-major, and must be symmetric; only the upper
 * triangle is read, so a caller that fills one triangle need not mirror it.
 */
export function jacobiEigen(matrix: Float64Array, n: number, sweeps = 100): Eigen {
  if (matrix.length < n * n) {
    throw new Error(`a ${n}x${n} matrix needs ${n * n} entries, got ${matrix.length}`);
  }
  // Working copy, fully symmetric.
  const a = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const value = matrix[i * n + j] as number;
      a[i * n + j] = value;
      a[j * n + i] = value;
    }
  }
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;

  for (let sweep = 0; sweep < sweeps; sweep++) {
    // The off-diagonal norm; Jacobi converges quadratically, so this falls
    // away fast once the matrix is nearly diagonal.
    let off = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) off += (a[i * n + j] as number) ** 2;
    }
    if (off <= Number.EPSILON * Number.EPSILON) break;

    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q] as number;
        if (apq === 0) continue;
        const app = a[p * n + p] as number;
        const aqq = a[q * n + q] as number;
        // The rotation that zeroes (p, q). Written with `theta` and the stable
        // branch for `t`, which avoids cancellation when the diagonal
        // difference dwarfs the off-diagonal term.
        const theta = (aqq - app) / (2 * apq);
        const t =
          theta >= 0
            ? 1 / (theta + Math.sqrt(1 + theta * theta))
            : -1 / (-theta + Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        for (let k = 0; k < n; k++) {
          const akp = a[k * n + p] as number;
          const akq = a[k * n + q] as number;
          a[k * n + p] = c * akp - s * akq;
          a[k * n + q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p * n + k] as number;
          const aqk = a[q * n + k] as number;
          a[p * n + k] = c * apk - s * aqk;
          a[q * n + k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k * n + p] as number;
          const vkq = v[k * n + q] as number;
          v[k * n + p] = c * vkp - s * vkq;
          v[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const order = Array.from({ length: n }, (_, i) => i).sort(
    (x, y) => (a[y * n + y] as number) - (a[x * n + x] as number),
  );
  const values = new Float64Array(n);
  const vectors = new Float64Array(n * n);
  for (const [rank, index] of order.entries()) {
    values[rank] = a[index * n + index] as number;
    // 🔴 Sign is fixed so the result is reproducible. An eigenvector is only
    // defined up to sign, and Jacobi's choice depends on the rotation order,
    // so two correct runs can disagree and a PCA projection can come out
    // mirrored. Making the largest-magnitude component positive removes that
    // freedom; it is the convention numpy users apply by hand when comparing.
    let largest = 0;
    for (let i = 1; i < n; i++) {
      if (Math.abs(v[i * n + index] as number) > Math.abs(v[largest * n + index] as number)) {
        largest = i;
      }
    }
    const flip = (v[largest * n + index] as number) < 0 ? -1 : 1;
    for (let i = 0; i < n; i++) vectors[i * n + rank] = flip * (v[i * n + index] as number);
  }
  return { values, vectors };
}
