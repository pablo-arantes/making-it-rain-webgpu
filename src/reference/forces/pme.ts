/**
 * Reciprocal-space PME, in double precision.
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
 * Rectangular boxes only, like the rest of the reference.
 */

import { ONE_4PI_EPS0 } from "../../engine/units.js";
import type { NonbondedForce, Vec3 } from "../../engine/system.js";
import { complexGrid, forward3d, inverse3d } from "../dft.js";
import { bsplineModuli, bsplineWeightsAndDerivatives } from "./bspline.js";
import { rectangularBox } from "./nonbonded.js";

/** The B-spline order OpenMM uses, and the engine with it. */
export const PME_ORDER = 5;

export interface ReciprocalResult {
  /** kJ/mol, excluding the self-energy term. */
  readonly energy: number;
  readonly grid: Vec3;
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
  coordinate: number,
  extent: number,
  gridSize: number,
): { base: number; weights: Float64Array; derivatives: Float64Array } {
  // Wrapped fractional coordinate, scaled to the grid.
  const fraction = coordinate / extent - Math.floor(coordinate / extent);
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
  const box = rectangularBox(boxVectors);
  const [nx, ny, nz] = gridSize as unknown as [number, number, number];
  const n = force.particleCount;
  const grid = complexGrid([nx, ny, nz]);

  // --- spread the charges -------------------------------------------------
  const splines: Array<ReturnType<typeof splineFor>[]> = [];
  for (let i = 0; i < n; i++) {
    const charge = force.charge[i] as number;
    const sx = splineFor(positions[i * 3] as number, box[0] as number, nx);
    const sy = splineFor(positions[i * 3 + 1] as number, box[1] as number, ny);
    const sz = splineFor(positions[i * 3 + 2] as number, box[2] as number, nz);
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

  forward3d(grid);

  // --- convolve with the Ewald kernel ------------------------------------
  const moduliX = bsplineModuli(nx, PME_ORDER);
  const moduliY = bsplineModuli(ny, PME_ORDER);
  const moduliZ = bsplineModuli(nz, PME_ORDER);
  const volume = (box[0] as number) * (box[1] as number) * (box[2] as number);
  // E = ONE_4PI_EPS0 * (2 pi / V) * sum_{k != 0} exp(-k^2/4 alpha^2)/k^2 |S(k)|^2,
  // which is the form tools/build_nonbonded_fixture.py checked against OpenMM's
  // exact Ewald. Here |S(k)|^2 is approximated by |Q(m)|^2 / |b(m)|^2.
  const prefactor = (ONE_4PI_EPS0 * 2 * Math.PI) / volume;
  const expScale = -1 / (4 * alpha * alpha);
  let energy = 0;

  for (let mx = 0; mx < nx; mx++) {
    // Frequencies above the midpoint are the negative ones.
    const kx = ((mx <= nx / 2 ? mx : mx - nx) * 2 * Math.PI) / (box[0] as number);
    for (let my = 0; my < ny; my++) {
      const ky = ((my <= ny / 2 ? my : my - ny) * 2 * Math.PI) / (box[1] as number);
      for (let mz = 0; mz < nz; mz++) {
        const at = ((mx * ny + my) * nz + mz) * 2;
        if (mx === 0 && my === 0 && mz === 0) {
          // The k = 0 term is the neutralising background; it is dropped, which
          // is the usual tin-foil boundary condition.
          grid.data[at] = 0;
          grid.data[at + 1] = 0;
          continue;
        }
        const kz = ((mz <= nz / 2 ? mz : mz - nz) * 2 * Math.PI) / (box[2] as number);
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

  if (forces === undefined) return { energy, grid: gridSize };

  // --- interpolate the potential gradient back onto the atoms ------------
  inverse3d(grid);
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
    forces[i * 3] = (forces[i * 3] as number) + (scale * fx * nx) / (box[0] as number);
    forces[i * 3 + 1] =
      (forces[i * 3 + 1] as number) + (scale * fy * ny) / (box[1] as number);
    forces[i * 3 + 2] =
      (forces[i * 3 + 2] as number) + (scale * fz * nz) / (box[2] as number);
  }
  return { energy, grid: gridSize };
}
