/**
 * Reciprocal-space PME, in double precision.
 *
 * Uses the mixed-radix transform in `src/reference/fft.ts`, which is itself
 * checked against the direct O(n^2) transform in `src/reference/dft.ts`. A 54^3
 * grid is 1.6e5 points, where the direct transform would be a few times 1e8
 * operations an axis.
 *
 * What this returns is the mesh sum alone. OpenMM's reciprocal force group is
 *
 *     reciprocalSpace(...).energy + selfEnergy(...) + netChargeCorrection(...)
 *
 * all three of which were pinned by measurement; see nonbonded.ts for the other
 * two.
 *
 * Smooth particle-mesh Ewald (Essmann et al. 1995): spread each charge onto a
 * mesh with B-splines, transform, multiply by the Ewald kernel divided by the
 * spline's own structure factor, transform back, and interpolate the potential
 * gradient to get forces.
 *
 * What OpenMM reports in its reciprocal force group is this plus the
 * self-energy term, which `selfEnergy` in nonbonded.ts supplies. The split was
 * measured, not assumed: tools/build_nonbonded_fixture.py reproduces OpenMM's
 * own two groups from first principles and agrees to 2e-9 kJ/mol.
 *
 * 🔴 Triclinic cells work, and the reason the conversion is small is worth
 * stating: the charge grid is laid out in **fractional** coordinates, so the
 * transform is the same transform whatever the cell shape -- `exp(ik·r)` is
 * `exp(2πi m·f)` by construction. Only three things depend on the shape: the
 * fractional coordinate the spline is evaluated at, the squared length of the
 * reciprocal vector in the convolution, and the chain rule that takes the
 * force from fractional back to Cartesian. For a rectangular box all three
 * collapse to what they were, so existing fixtures are unchanged bit for bit.
 *
 * The risk was never the algebra, it was having nothing to check it against:
 * the direct and reciprocal terms cancel to four or five digits, so a wrong
 * reciprocal lattice reads as a slightly different total rather than as an
 * error. `src/reference/forces/ewald.ts` is the answer -- an exact Ewald sum
 * with no mesh, no spline and no transform, which shares nothing with this but
 * the physics.
 */

import { ONE_4PI_EPS0 } from "../../engine/units.js";
import type { NonbondedForce, Vec3 } from "../../engine/system.js";
import { complexGrid, forward3d, inverse3d } from "../dft.js";
import { fft3d, planFft } from "../fft.js";
import { bsplineModuli, bsplineWeightsAndDerivatives } from "./bspline.js";
import { usableBox } from "./nonbonded.js";
import {
  boxVolume,
  fractionalOf,
  inverseBox,
  reciprocalVector,
} from "../triclinic.js";

/** The B-spline order OpenMM uses, and the engine with it. */
export const PME_ORDER = 5;

export interface ReciprocalResult {
  /** kJ/mol, excluding the self-energy and net-charge terms. */
  readonly energy: number;
  readonly grid: Vec3;
  /** Whether the mixed-radix transform could be used, or the direct one was. */
  readonly smoothGrid: boolean;
}

/**
 * Where an atom sits on the mesh, and the spline weights around it.
 *
 * The grid points an atom touches are `floor(u) - order + 1 + k` for
 * k = 0..order-1, wrapped. That offset convention has to match between
 * spreading and interpolation or the forces come out shifted by a grid cell,
 * which is why both go through this one function.
 */
function splineFor(
  fractional: number,
  gridSize: number,
): { base: number; weights: Float64Array; derivatives: Float64Array } {
  // The caller supplies the fractional coordinate, which is `r / L` for a
  // rectangular box and `r · B⁻¹` in general; this only has to wrap it.
  const fraction = fractional - Math.floor(fractional);
  const scaled = fraction * gridSize;
  const cell = Math.floor(scaled);
  const { weights, derivatives } = bsplineWeightsAndDerivatives(scaled - cell, PME_ORDER);
  return { base: cell - PME_ORDER + 1, weights, derivatives };
}

/**
 * Reciprocal-space energy and, optionally, forces.
 *
 * `forces` is accumulated into, in kJ/mol/nm.
 */
export function reciprocalSpace(
  force: NonbondedForce,
  positions: Float64Array,
  boxVectors: readonly [Vec3, Vec3, Vec3],
  alpha: number,
  gridSize: Vec3,
  forces?: Float64Array,
): ReciprocalResult {
  const box = usableBox(boxVectors, force.cutoff);
  const inverse = inverseBox(box);
  const [nx, ny, nz] = gridSize as unknown as [number, number, number];
  const n = force.particleCount;
  const grid = complexGrid([nx, ny, nz]);
  // The mixed-radix transform where it applies, the direct one where it does
  // not. This is not a performance fallback, it is a correctness requirement:
  // OpenMM's *Reference* platform uses the exact minimum grid, which is often
  // not smooth -- 53 for the solvated protein-ligand fixture and 22 for the
  // committed one, both of which have a prime factor no radix covers. Its GPU
  // platforms round up, as this engine does, and comparing against the
  // Reference platform means being able to run at its grid.
  const smooth = [nx, ny, nz].every((n) => {
    try {
      planFft(n);
      return true;
    } catch {
      return false;
    }
  });
  const scratch = smooth ? new Float64Array(nx * ny * nz * 2) : undefined;
  const transform = (sign: number): void => {
    if (smooth) fft3d(grid.data, [nx, ny, nz], sign, scratch);
    else if (sign < 0) forward3d(grid);
    else inverse3d(grid);
  };

  // --- spread the charges -------------------------------------------------
  const splines: Array<ReturnType<typeof splineFor>[]> = [];
  for (let i = 0; i < n; i++) {
    const charge = force.charge[i] as number;
    const fractional = fractionalOf(
      positions[i * 3] as number,
      positions[i * 3 + 1] as number,
      positions[i * 3 + 2] as number,
      inverse,
    );
    const sx = splineFor(fractional[0] as number, nx);
    const sy = splineFor(fractional[1] as number, ny);
    const sz = splineFor(fractional[2] as number, nz);
    splines.push([sx, sy, sz]);
    if (charge === 0) continue;
    for (let a = 0; a < PME_ORDER; a++) {
      const gx = (((sx.base + a) % nx) + nx) % nx;
      const wx = charge * (sx.weights[a] as number);
      for (let b = 0; b < PME_ORDER; b++) {
        const gy = (((sy.base + b) % ny) + ny) % ny;
        const wxy = wx * (sy.weights[b] as number);
        for (let c = 0; c < PME_ORDER; c++) {
          const gz = (((sz.base + c) % nz) + nz) % nz;
          const at = ((gx * ny + gy) * nz + gz) * 2;
          grid.data[at] = (grid.data[at] as number) + wxy * (sz.weights[c] as number);
        }
      }
    }
  }

  transform(-1);

  // --- convolve with the Ewald kernel ------------------------------------
  const moduliX = bsplineModuli(nx, PME_ORDER);
  const moduliY = bsplineModuli(ny, PME_ORDER);
  const moduliZ = bsplineModuli(nz, PME_ORDER);
  const volume = boxVolume(box);
  // E = ONE_4PI_EPS0 * (2 pi / V) * sum_{k != 0} exp(-k^2/4 alpha^2)/k^2 |S(k)|^2,
  // which is the form tools/build_nonbonded_fixture.py checked against OpenMM's
  // exact Ewald. Here |S(k)|^2 is approximated by |Q(m)|^2 / |b(m)|^2.
  const prefactor = (ONE_4PI_EPS0 * 2 * Math.PI) / volume;
  const expScale = -1 / (4 * alpha * alpha);
  let energy = 0;

  for (let mx = 0; mx < nx; mx++) {
    // Frequencies above the midpoint are the negative ones.
    const ma = mx <= nx / 2 ? mx : mx - nx;
    for (let my = 0; my < ny; my++) {
      const mb = my <= ny / 2 ? my : my - ny;
      for (let mz = 0; mz < nz; mz++) {
        const at = ((mx * ny + my) * nz + mz) * 2;
        if (mx === 0 && my === 0 && mz === 0) {
          // The k = 0 term is the neutralising background; it is dropped, which
          // is the usual tin-foil boundary condition.
          grid.data[at] = 0;
          grid.data[at + 1] = 0;
          continue;
        }
        const mc = mz <= nz / 2 ? mz : mz - nz;
        // 🔴 `k` is a *vector* on the reciprocal lattice, not three independent
        // per-axis terms. For a rectangular box it is `2π m_u / L_u` per axis
        // and this is the same number; for a skewed one the components mix, and
        // treating them independently is the mistake that would read as a small
        // shift in the total rather than as a wrong answer.
        const k = reciprocalVector(ma, mb, mc, inverse);
        const kx = k[0] as number;
        const ky = k[1] as number;
        const kz = k[2] as number;
        const kSquared = kx * kx + ky * ky + kz * kz;
        const modulus =
          (moduliX[mx] as number) * (moduliY[my] as number) * (moduliZ[mz] as number);
        const factor = (prefactor * Math.exp(kSquared * expScale)) / (kSquared * modulus);
        const re = grid.data[at] as number;
        const im = grid.data[at + 1] as number;
        energy += factor * (re * re + im * im);
        // Twice the factor, because the energy is quadratic in the grid and the
        // force needs dE/dQ.
        grid.data[at] = 2 * factor * re;
        grid.data[at + 1] = 2 * factor * im;
      }
    }
  }

  if (forces === undefined) return { energy, grid: gridSize, smoothGrid: smooth };

  // --- interpolate the potential gradient back onto the atoms ------------
  transform(1);
  for (let i = 0; i < n; i++) {
    const charge = force.charge[i] as number;
    if (charge === 0) continue;
    const [sx, sy, sz] = splines[i] as ReturnType<typeof splineFor>[];
    let fx = 0;
    let fy = 0;
    let fz = 0;
    for (let a = 0; a < PME_ORDER; a++) {
      const gx = (((sx!.base + a) % nx) + nx) % nx;
      const wa = sx!.weights[a] as number;
      const da = sx!.derivatives[a] as number;
      for (let b = 0; b < PME_ORDER; b++) {
        const gy = (((sy!.base + b) % ny) + ny) % ny;
        const wb = sy!.weights[b] as number;
        const db = sy!.derivatives[b] as number;
        for (let c = 0; c < PME_ORDER; c++) {
          const gz = (((sz!.base + c) % nz) + nz) % nz;
          const wc = sz!.weights[c] as number;
          const dc = sz!.derivatives[c] as number;
          const potential = grid.data[((gx * ny + gy) * nz + gz) * 2] as number;
          // dE/dr = q * sum_grid d(weights)/dr * potential, and d/dr = (grid
          // points per nm) * d/dw.
          fx += potential * da * wb * wc;
          fy += potential * wa * db * wc;
          fz += potential * wa * wb * dc;
        }
      }
    }
    // No extra factor: the two the convolution carried *is* dE/dQ.
    //   E = sum_m f(m) |Q(m)|^2, so dE/dQ(n) = 2 Re[IDFT(f Q)](n),
    // and F_i = -q_i sum_n dW/dr_i * dE/dQ(n). A half here made every
    // reciprocal-space force exactly twice too small, which the numerical
    // derivative caught as a relative error of 0.5.
    const scale = -charge;
    // The spline derivatives are with respect to the *fractional* coordinate,
    // so the chain rule brings them back: `dE/dr_v = Σ_u dE/df_u · B⁻¹[v][u]`.
    // For a rectangular box `B⁻¹` is diagonal and this is the division by the
    // edge length it replaced.
    const df: [number, number, number] = [scale * fx * nx, scale * fy * ny, scale * fz * nz];
    for (let axis = 0; axis < 3; axis++) {
      forces[i * 3 + axis] =
        (forces[i * 3 + axis] as number) +
        (df[0] as number) * ((inverse[axis] as Vec3)[0] as number) +
        (df[1] as number) * ((inverse[axis] as Vec3)[1] as number) +
        (df[2] as number) * ((inverse[axis] as Vec3)[2] as number);
    }
  }
  return { energy, grid: gridSize, smoothGrid: smooth };
}
