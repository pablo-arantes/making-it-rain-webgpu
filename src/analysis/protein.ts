/**
 * The protein analyses the notebook plots: Cα RMSD, radius of gyration, RMSF,
 * 2D RMSD, PCA and the dynamic cross-correlation matrix.
 *
 * `docs/NOTEBOOK_PARITY.md` maps each one to the pytraj call it replaces (cells
 * 38 to 46). Every one of them operates on a *subset* of atoms, almost always
 * the Cα trace, and almost all of them need the frames superposed first — so
 * the two primitives here are atom selection and best-fit superposition, and
 * the analyses are short on top of them.
 *
 * 🔴 **Alignment is explicit, because cpptraj's is not.** `pt.rmsf` and
 * `pytraj.matrix.correl` do no fitting of their own: they assume the trajectory
 * was aligned by a previous step, and give meaningless answers on a trajectory
 * that was not. The notebook aligns before the cross-correlation (cell 46) and
 * does not before the RMSF (cell 42), which leaves cell 42 depending on
 * whatever pytraj's internal state happened to be. Rather than reproduce an
 * accident, every function here takes the frames it is given and says in its
 * own documentation whether it superposes. Where the notebook's intent is
 * clear — a fluctuation is about an aligned mean, or it is measuring the
 * molecule tumbling — this aligns and says so.
 *
 * Everything works on a flat `Float32Array` of `frames * atoms * 3` in nm,
 * which is what `readDcd` returns.
 */

import { jacobiEigen } from "./eigen.js";
import type { PdbTopology } from "./pdb.js";

/** Residue names the Cα selection treats as protein. */
const NOT_PROTEIN = new Set([
  "HOH", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "LI", "RB", "CS", "BR", "F", "I",
]);

/**
 * The Cα atoms, in topology order.
 *
 * 🔴 The name test alone is wrong, and this is the `CA` ambiguity again: a
 * calcium ion is also called `CA`, and its residue is called `CA` too. Selecting
 * by name would put a calcium in the Cα trace, which tilts an RMSD and shifts
 * a radius of gyration. The element has to agree as well.
 */
export function caIndices(topology: PdbTopology): readonly number[] {
  const found: number[] = [];
  for (const [index, atom] of topology.atoms.entries()) {
    if (atom.name !== "CA") continue;
    if (atom.element.toUpperCase() !== "C") continue;
    if (NOT_PROTEIN.has(atom.residueName)) continue;
    found.push(index);
  }
  return found;
}

/** Pull one frame's selected atoms out as a dense `3 * k` array. */
export function selectFrame(
  positions: Float32Array | Float64Array,
  atomCount: number,
  frame: number,
  indices: readonly number[],
): Float64Array {
  const out = new Float64Array(indices.length * 3);
  const base = frame * atomCount * 3;
  for (const [slot, atom] of indices.entries()) {
    out[slot * 3] = positions[base + atom * 3] as number;
    out[slot * 3 + 1] = positions[base + atom * 3 + 1] as number;
    out[slot * 3 + 2] = positions[base + atom * 3 + 2] as number;
  }
  return out;
}

/** Subtract the centroid in place, returning it. */
export function centre(coordinates: Float64Array): readonly [number, number, number] {
  const n = coordinates.length / 3;
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < n; i++) {
    cx += coordinates[i * 3] as number;
    cy += coordinates[i * 3 + 1] as number;
    cz += coordinates[i * 3 + 2] as number;
  }
  cx /= n;
  cy /= n;
  cz /= n;
  for (let i = 0; i < n; i++) {
    coordinates[i * 3] = (coordinates[i * 3] as number) - cx;
    coordinates[i * 3 + 1] = (coordinates[i * 3 + 1] as number) - cy;
    coordinates[i * 3 + 2] = (coordinates[i * 3 + 2] as number) - cz;
  }
  return [cx, cy, cz];
}

export interface Superposition {
  /** nm. */
  readonly rmsd: number;
  /** Row-major 3x3, rotating the centred mobile set onto the centred reference. */
  readonly rotation: Float64Array;
}

/**
 * Best-fit superposition of `mobile` onto `reference`, both centred in place.
 *
 * The quaternion method (Horn; Kearsley), not SVD: the optimal rotation is the
 * eigenvector of a 4x4 symmetric matrix built from the 3x3 correlation matrix,
 * and `jacobiEigen` already solves that. It also hands back the RMSD without
 * applying the rotation, from
 *
 *     rmsd = sqrt(max(0, (E0 - 2 * lambda_max) / n))
 *
 * where `E0` is the summed squared norms. Clamping at zero matters: for two
 * identical structures the expression is a difference of equal quantities and
 * lands a few 1e-16 either side, so the square root of a tiny negative is a
 * real possibility and would come back NaN.
 */
export function superpose(mobile: Float64Array, reference: Float64Array): Superposition {
  const n = mobile.length / 3;
  if (reference.length !== mobile.length) {
    throw new Error(
      `superposing ${n} atoms onto ${reference.length / 3}: the selections differ`,
    );
  }
  centre(mobile);
  centre(reference);

  // R[i][j] = sum_k mobile_k[i] * reference_k[j]
  const r = new Float64Array(9);
  let e0 = 0;
  for (let k = 0; k < n; k++) {
    const mx = mobile[k * 3] as number;
    const my = mobile[k * 3 + 1] as number;
    const mz = mobile[k * 3 + 2] as number;
    const rx = reference[k * 3] as number;
    const ry = reference[k * 3 + 1] as number;
    const rz = reference[k * 3 + 2] as number;
    e0 += mx * mx + my * my + mz * mz + rx * rx + ry * ry + rz * rz;
    r[0] = (r[0] as number) + mx * rx;
    r[1] = (r[1] as number) + mx * ry;
    r[2] = (r[2] as number) + mx * rz;
    r[3] = (r[3] as number) + my * rx;
    r[4] = (r[4] as number) + my * ry;
    r[5] = (r[5] as number) + my * rz;
    r[6] = (r[6] as number) + mz * rx;
    r[7] = (r[7] as number) + mz * ry;
    r[8] = (r[8] as number) + mz * rz;
  }
  const r11 = r[0] as number;
  const r12 = r[1] as number;
  const r13 = r[2] as number;
  const r21 = r[3] as number;
  const r22 = r[4] as number;
  const r23 = r[5] as number;
  const r31 = r[6] as number;
  const r32 = r[7] as number;
  const r33 = r[8] as number;

  const f = new Float64Array(16);
  const set = (i: number, j: number, value: number): void => {
    f[i * 4 + j] = value;
    f[j * 4 + i] = value;
  };
  set(0, 0, r11 + r22 + r33);
  set(1, 1, r11 - r22 - r33);
  set(2, 2, -r11 + r22 - r33);
  set(3, 3, -r11 - r22 + r33);
  set(0, 1, r23 - r32);
  set(0, 2, r31 - r13);
  set(0, 3, r12 - r21);
  set(1, 2, r12 + r21);
  set(1, 3, r13 + r31);
  set(2, 3, r23 + r32);

  const { values, vectors } = jacobiEigen(f, 4);
  const lambda = values[0] as number;
  const rmsd = Math.sqrt(Math.max(0, (e0 - 2 * lambda) / n));

  const q0 = vectors[0 * 4 + 0] as number;
  const q1 = vectors[1 * 4 + 0] as number;
  const q2 = vectors[2 * 4 + 0] as number;
  const q3 = vectors[3 * 4 + 0] as number;
  const rotation = Float64Array.from([
    q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3,
    2 * (q1 * q2 - q0 * q3),
    2 * (q1 * q3 + q0 * q2),
    2 * (q1 * q2 + q0 * q3),
    q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3,
    2 * (q2 * q3 - q0 * q1),
    2 * (q1 * q3 - q0 * q2),
    2 * (q2 * q3 + q0 * q1),
    q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3,
  ]);
  return { rmsd, rotation };
}

/** Apply a 3x3 rotation to every atom of a centred set, in place. */
export function rotate(coordinates: Float64Array, rotation: Float64Array): void {
  const n = coordinates.length / 3;
  for (let i = 0; i < n; i++) {
    const x = coordinates[i * 3] as number;
    const y = coordinates[i * 3 + 1] as number;
    const z = coordinates[i * 3 + 2] as number;
    coordinates[i * 3] =
      (rotation[0] as number) * x + (rotation[1] as number) * y + (rotation[2] as number) * z;
    coordinates[i * 3 + 1] =
      (rotation[3] as number) * x + (rotation[4] as number) * y + (rotation[5] as number) * z;
    coordinates[i * 3 + 2] =
      (rotation[6] as number) * x + (rotation[7] as number) * y + (rotation[8] as number) * z;
  }
}

export interface Trajectory {
  readonly frameCount: number;
  readonly particleCount: number;
  /** nm, `frameCount * particleCount * 3`. */
  readonly positions: Float32Array | Float64Array;
}

/**
 * Cα RMSD against one frame, superposing each frame first.
 *
 * `pt.rmsd(traj, ref=0, mask='@CA')`. The fit and the measurement use the same
 * atoms, which is what pytraj does with a single mask.
 */
export function rmsdSeries(
  trajectory: Trajectory,
  indices: readonly number[],
  referenceFrame = 0,
): Float64Array {
  const reference = selectFrame(
    trajectory.positions,
    trajectory.particleCount,
    referenceFrame,
    indices,
  );
  centre(reference);
  const out = new Float64Array(trajectory.frameCount);
  for (let frame = 0; frame < trajectory.frameCount; frame++) {
    const mobile = selectFrame(
      trajectory.positions,
      trajectory.particleCount,
      frame,
      indices,
    );
    out[frame] = superpose(mobile, Float64Array.from(reference)).rmsd;
  }
  return out;
}

/**
 * Radius of gyration per frame, over the selection.
 *
 * `pt.radgyr(mask='@CA')`. cpptraj mass-weights, and this does not — which is
 * the same number for a Cα selection, because every Cα has the same mass. The
 * difference would appear only on a selection mixing elements, and the
 * notebook never takes one.
 */
export function radiusOfGyration(
  trajectory: Trajectory,
  indices: readonly number[],
): Float64Array {
  const out = new Float64Array(trajectory.frameCount);
  for (let frame = 0; frame < trajectory.frameCount; frame++) {
    const coordinates = selectFrame(
      trajectory.positions,
      trajectory.particleCount,
      frame,
      indices,
    );
    centre(coordinates);
    let total = 0;
    for (let i = 0; i < indices.length; i++) {
      total +=
        (coordinates[i * 3] as number) ** 2 +
        (coordinates[i * 3 + 1] as number) ** 2 +
        (coordinates[i * 3 + 2] as number) ** 2;
    }
    out[frame] = Math.sqrt(total / indices.length);
  }
  return out;
}

/**
 * Every frame's selection, superposed onto the first, as one array.
 *
 * The shared first step of RMSF, PCA and the cross-correlation matrix: all
 * three measure deviations from a mean, and a mean over unaligned frames is
 * dominated by the molecule tumbling rather than by anything internal.
 */
export function alignedCoordinates(
  trajectory: Trajectory,
  indices: readonly number[],
  referenceFrame = 0,
): Float64Array {
  const k = indices.length;
  const reference = selectFrame(
    trajectory.positions,
    trajectory.particleCount,
    referenceFrame,
    indices,
  );
  centre(reference);
  const out = new Float64Array(trajectory.frameCount * k * 3);
  for (let frame = 0; frame < trajectory.frameCount; frame++) {
    const mobile = selectFrame(
      trajectory.positions,
      trajectory.particleCount,
      frame,
      indices,
    );
    const { rotation } = superpose(mobile, Float64Array.from(reference));
    rotate(mobile, rotation);
    out.set(mobile, frame * k * 3);
  }
  return out;
}

/** The mean structure of an aligned coordinate set. */
export function meanStructure(aligned: Float64Array, frames: number): Float64Array {
  const size = aligned.length / frames;
  const mean = new Float64Array(size);
  for (let frame = 0; frame < frames; frame++) {
    for (let i = 0; i < size; i++) {
      mean[i] = (mean[i] as number) + (aligned[frame * size + i] as number);
    }
  }
  for (let i = 0; i < size; i++) mean[i] = (mean[i] as number) / frames;
  return mean;
}

/**
 * Per-atom root mean square fluctuation about the mean structure, nm.
 *
 * `pt.rmsf('@CA')`. Frames are superposed onto the reference first — cpptraj
 * does not do that for you, and without it this measures tumbling.
 */
export function rmsf(
  trajectory: Trajectory,
  indices: readonly number[],
  referenceFrame = 0,
): Float64Array {
  const aligned = alignedCoordinates(trajectory, indices, referenceFrame);
  const frames = trajectory.frameCount;
  const mean = meanStructure(aligned, frames);
  const out = new Float64Array(indices.length);
  for (let atom = 0; atom < indices.length; atom++) {
    let total = 0;
    for (let frame = 0; frame < frames; frame++) {
      const base = frame * indices.length * 3 + atom * 3;
      total +=
        ((aligned[base] as number) - (mean[atom * 3] as number)) ** 2 +
        ((aligned[base + 1] as number) - (mean[atom * 3 + 1] as number)) ** 2 +
        ((aligned[base + 2] as number) - (mean[atom * 3 + 2] as number)) ** 2;
    }
    out[atom] = Math.sqrt(total / frames);
  }
  return out;
}

/**
 * The all-pairs best-fit RMSD matrix, nm, row-major and symmetric.
 *
 * `pt.pairwise_rmsd(mask='@CA')`. Each pair is superposed independently, so
 * this is O(frames^2) superpositions — 100 frames is 4950 of them, which is
 * milliseconds, and 10,000 frames would not be.
 */
export function pairwiseRmsd(
  trajectory: Trajectory,
  indices: readonly number[],
): Float64Array {
  const frames = trajectory.frameCount;
  const selections: Float64Array[] = [];
  for (let frame = 0; frame < frames; frame++) {
    const coordinates = selectFrame(
      trajectory.positions,
      trajectory.particleCount,
      frame,
      indices,
    );
    centre(coordinates);
    selections.push(coordinates);
  }
  const out = new Float64Array(frames * frames);
  for (let i = 0; i < frames; i++) {
    for (let j = i + 1; j < frames; j++) {
      const { rmsd } = superpose(
        Float64Array.from(selections[i] as Float64Array),
        Float64Array.from(selections[j] as Float64Array),
      );
      out[i * frames + j] = rmsd;
      out[j * frames + i] = rmsd;
    }
  }
  return out;
}

export interface Pca {
  /** Descending, nm^2. One per coordinate, so `3 * atoms` of them. */
  readonly eigenvalues: Float64Array;
  /** Column `k` is component `k`: `vectors[i * size + k]`. */
  readonly eigenvectors: Float64Array;
  /** `frames * components`, the projection of each frame onto each component. */
  readonly projections: Float64Array;
  readonly components: number;
}

/**
 * Principal components of the aligned Cα coordinates.
 *
 * `pt.pca(fit=True, ref=0, mask='@CA', n_vecs=2)`: superpose onto the
 * reference, take the covariance of the `3 * atoms` coordinates about their
 * mean, and project each frame onto the leading eigenvectors.
 *
 * The covariance is normalised by the frame count, not by `frames - 1`, which
 * is cpptraj's convention.
 */
export function pca(
  trajectory: Trajectory,
  indices: readonly number[],
  components = 2,
  referenceFrame = 0,
): Pca {
  const frames = trajectory.frameCount;
  const size = indices.length * 3;
  const aligned = alignedCoordinates(trajectory, indices, referenceFrame);
  const mean = meanStructure(aligned, frames);

  const covariance = new Float64Array(size * size);
  for (let frame = 0; frame < frames; frame++) {
    const base = frame * size;
    for (let i = 0; i < size; i++) {
      const di = (aligned[base + i] as number) - (mean[i] as number);
      if (di === 0) continue;
      for (let j = i; j < size; j++) {
        covariance[i * size + j] =
          (covariance[i * size + j] as number) +
          di * ((aligned[base + j] as number) - (mean[j] as number));
      }
    }
  }
  for (let i = 0; i < size; i++) {
    for (let j = i; j < size; j++) {
      const value = (covariance[i * size + j] as number) / frames;
      covariance[i * size + j] = value;
      covariance[j * size + i] = value;
    }
  }

  const { values, vectors } = jacobiEigen(covariance, size);
  const wanted = Math.min(components, size);
  const projections = new Float64Array(frames * wanted);
  for (let frame = 0; frame < frames; frame++) {
    const base = frame * size;
    for (let k = 0; k < wanted; k++) {
      let total = 0;
      for (let i = 0; i < size; i++) {
        total +=
          ((aligned[base + i] as number) - (mean[i] as number)) *
          (vectors[i * size + k] as number);
      }
      projections[frame * wanted + k] = total;
    }
  }
  return { eigenvalues: values, eigenvectors: vectors, projections, components: wanted };
}

/**
 * The dynamic cross-correlation matrix over the selection, row-major.
 *
 * `pt.align(mask='@CA', ref=0)` then `pytraj.matrix.correl('@CA')`:
 *
 *     C_ij = <dr_i . dr_j> / sqrt(<|dr_i|^2> <|dr_j|^2>)
 *
 * with `dr` the displacement from the mean structure. The diagonal is exactly
 * 1 and the range is -1 to 1, which is why the notebook's colour scale is
 * symmetric about zero.
 */
export function crossCorrelation(
  trajectory: Trajectory,
  indices: readonly number[],
  referenceFrame = 0,
): Float64Array {
  const frames = trajectory.frameCount;
  const k = indices.length;
  const aligned = alignedCoordinates(trajectory, indices, referenceFrame);
  const mean = meanStructure(aligned, frames);

  // <dr_i . dr_j>, and the diagonal doubles as <|dr_i|^2>.
  const dot = new Float64Array(k * k);
  for (let frame = 0; frame < frames; frame++) {
    const base = frame * k * 3;
    const d = new Float64Array(k * 3);
    for (let i = 0; i < k * 3; i++) {
      d[i] = (aligned[base + i] as number) - (mean[i] as number);
    }
    for (let i = 0; i < k; i++) {
      for (let j = i; j < k; j++) {
        dot[i * k + j] =
          (dot[i * k + j] as number) +
          (d[i * 3] as number) * (d[j * 3] as number) +
          (d[i * 3 + 1] as number) * (d[j * 3 + 1] as number) +
          (d[i * 3 + 2] as number) * (d[j * 3 + 2] as number);
      }
    }
  }

  const out = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = i; j < k; j++) {
      const norm = Math.sqrt(
        (dot[i * k + i] as number) * (dot[j * k + j] as number),
      );
      const value = norm === 0 ? 0 : (dot[i * k + j] as number) / norm;
      out[i * k + j] = value;
      out[j * k + i] = value;
    }
  }
  return out;
}

/**
 * A Gaussian kernel density estimate, for the distribution plots.
 *
 * The notebook draws seaborn KDEs beside the RMSD, radius of gyration and PCA
 * traces (cells 39, 41, 45). seaborn's default bandwidth is Scott's rule,
 * `n^(-1/5)` times the standard deviation for one dimension, and that is what
 * this uses — a different bandwidth gives a visibly different curve from the
 * same data, so it is worth matching rather than picking.
 */
export function kernelDensity(
  samples: ArrayLike<number>,
  points = 200,
): { readonly x: Float64Array; readonly y: Float64Array; readonly bandwidth: number } {
  const n = samples.length;
  if (n === 0) return { x: new Float64Array(0), y: new Float64Array(0), bandwidth: 0 };
  let mean = 0;
  for (let i = 0; i < n; i++) mean += samples[i] as number;
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += ((samples[i] as number) - mean) ** 2;
  // The sample standard deviation, which is what Scott's rule is stated for.
  const sd = Math.sqrt(variance / Math.max(1, n - 1));
  const bandwidth = sd * Math.pow(n, -1 / 5);

  let low = Infinity;
  let high = -Infinity;
  for (let i = 0; i < n; i++) {
    const value = samples[i] as number;
    if (value < low) low = value;
    if (value > high) high = value;
  }
  // Three bandwidths of margin, so the curve reaches the axis rather than
  // being cut off at the extreme samples.
  const margin = 3 * (bandwidth > 0 ? bandwidth : Math.max(1e-9, (high - low) || 1));
  low -= margin;
  high += margin;

  const x = new Float64Array(points);
  const y = new Float64Array(points);
  if (bandwidth === 0) {
    // Every sample identical: a density is a spike, and dividing by zero
    // bandwidth would give NaN across the board.
    for (let i = 0; i < points; i++) {
      x[i] = low + ((high - low) * i) / (points - 1);
      y[i] = 0;
    }
    return { x, y, bandwidth };
  }
  const scale = 1 / (n * bandwidth * Math.sqrt(2 * Math.PI));
  for (let i = 0; i < points; i++) {
    const at = low + ((high - low) * i) / (points - 1);
    x[i] = at;
    let total = 0;
    for (let j = 0; j < n; j++) {
      const z = (at - (samples[j] as number)) / bandwidth;
      total += Math.exp(-0.5 * z * z);
    }
    y[i] = total * scale;
  }
  return { x, y, bandwidth };
}

/** Comma-separated values with a header row, for a per-frame series. */
export function seriesCsv(
  columns: readonly { readonly name: string; readonly values: ArrayLike<number> }[],
): string {
  const rows = columns.reduce((most, column) => Math.max(most, column.values.length), 0);
  const lines = [columns.map((column) => column.name).join(",")];
  for (let row = 0; row < rows; row++) {
    lines.push(
      columns
        .map((column) =>
          row < column.values.length ? String(column.values[row] as number) : "",
        )
        .join(","),
    );
  }
  return lines.join("\n") + "\n";
}

/** A square matrix as CSV, for the 2D RMSD and cross-correlation grids. */
export function matrixCsv(matrix: ArrayLike<number>, size: number): string {
  const lines: string[] = [];
  for (let i = 0; i < size; i++) {
    const row: string[] = [];
    for (let j = 0; j < size; j++) row.push(String(matrix[i * size + j] as number));
    lines.push(row.join(","));
  }
  return lines.join("\n") + "\n";
}
