/**
 * The reciprocal-space Ewald sum, exactly, with no mesh and no FFT.
 *
 * 🔴 This exists to be the oracle for a *triclinic* PME, and the plan named it
 * before the problem arrived: "keep an Ewald-sum (no FFT) path as the
 * differential reference for small systems, where the exact answer is
 * computable". PME is the single biggest numerical risk in this engine — the
 * direct and reciprocal terms cancel to four or five digits, so a wrong
 * reciprocal lattice does not look wrong, it looks like a slightly different
 * total — and the triclinic reciprocal lattice is the part of the conversion
 * that cannot be checked by inspection.
 *
 * It is O(N · K³) and is for fixtures, never for a run. What makes it a usable
 * oracle is that it shares nothing with `reciprocalSpace` but the physics: no
 * B-spline, no grid, no transform, no index arithmetic. If the two agree on a
 * skewed cell, the mesh path's reciprocal vectors are right.
 *
 * The convention is OpenMM's: energies in kJ/mol, lengths in nm, charges in
 * elementary charge, and the self and net-charge terms excluded — they are
 * volume- or charge-dependent constants that `potential.ts` adds once, so
 * including them here would double them.
 */

import { ONE_4PI_EPS0 } from "../../engine/units.js";
import type { Vec3 } from "../../engine/system.js";
import {
  boxVolume,
  inverseBox,
  maximumCutoff,
  reciprocalVector,
  type BoxVectors,
} from "../triclinic.js";

export interface EwaldReciprocal {
  /** kJ/mol, the reciprocal-space energy without the self or net-charge terms. */
  readonly energy: number;
  /** How many non-zero lattice vectors were summed. */
  readonly vectors: number;
  /** The mesh half-width actually used per axis. */
  readonly halfWidths: readonly [number, number, number];
}

/**
 * How many reciprocal vectors to sum along each axis.
 *
 * The summand decays as `exp(-k²/4α²)`, so a half-width of `n` contributes at
 * worst `exp(-(π n / (α L))²)` — the bound is per *perpendicular width*, not per
 * edge, for the same reason the cutoff is. `tolerance` is that worst-case
 * factor; 1e-10 is far tighter than PME's own error and makes the sum an oracle
 * rather than another approximation.
 */
export function ewaldHalfWidths(
  box: BoxVectors,
  alpha: number,
  tolerance = 1e-10,
): [number, number, number] {
  const inverse = inverseBox(box);
  const target = Math.sqrt(-Math.log(tolerance));
  const widths: [number, number, number] = [1, 1, 1];
  for (let axis = 0; axis < 3; axis++) {
    // |k| for one step along this axis, from the reciprocal vector itself, so a
    // skewed cell is handled by the algebra rather than by an assumption.
    const step = reciprocalVector(
      axis === 0 ? 1 : 0,
      axis === 1 ? 1 : 0,
      axis === 2 ? 1 : 0,
      inverse,
    );
    const magnitude = Math.hypot(step[0] as number, step[1] as number, step[2] as number);
    widths[axis] = Math.max(1, Math.ceil((2 * alpha * target) / magnitude));
  }
  return widths;
}

/**
 * The reciprocal-space energy and, optionally, its forces.
 *
 * `E = (1/2ε₀V) Σ_{k≠0} exp(-k²/4α²)/k² |S(k)|²` with `S(k) = Σ_j q_j e^{ik·r_j}`,
 * written with OpenMM's `ONE_4PI_EPS0` so the units match the rest of the
 * reference: `1/(2 ε₀ V) = 4π · ONE_4PI_EPS0 / (2 V) = 2π · ONE_4PI_EPS0 / V`.
 */
export function ewaldReciprocal(
  charges: readonly number[] | Float64Array,
  positions: Float64Array,
  box: BoxVectors,
  alpha: number,
  options: { readonly tolerance?: number; readonly forces?: Float64Array } = {},
): EwaldReciprocal {
  const inverse = inverseBox(box);
  const volume = boxVolume(box);
  const halfWidths = ewaldHalfWidths(box, alpha, options.tolerance);
  const n = positions.length / 3;
  const prefactor = (2 * Math.PI * ONE_4PI_EPS0) / volume;
  const quarterOverAlphaSquared = 1 / (4 * alpha * alpha);

  let energy = 0;
  let vectors = 0;
  // Half the lattice, doubled, because S(-k) = conj(S(k)) and the summand is
  // even in k. Enumerated so that exactly one of each +/- pair is visited: ma
  // positive, or ma zero and mb positive, or both zero and mc positive.
  for (let ma = 0; ma <= (halfWidths[0] as number); ma++) {
    const mbFrom = ma === 0 ? 0 : -(halfWidths[1] as number);
    for (let mb = mbFrom; mb <= (halfWidths[1] as number); mb++) {
      const mcFrom = ma === 0 && mb === 0 ? 1 : -(halfWidths[2] as number);
      for (let mc = mcFrom; mc <= (halfWidths[2] as number); mc++) {
        const k = reciprocalVector(ma, mb, mc, inverse);
        const kx = k[0] as number;
        const ky = k[1] as number;
        const kz = k[2] as number;
        const kSquared = kx * kx + ky * ky + kz * kz;
        if (kSquared === 0) continue;
        const weight = Math.exp(-kSquared * quarterOverAlphaSquared) / kSquared;
        if (weight === 0) continue;
        vectors++;

        let real = 0;
        let imaginary = 0;
        for (let i = 0; i < n; i++) {
          const phase =
            kx * (positions[i * 3] as number) +
            ky * (positions[i * 3 + 1] as number) +
            kz * (positions[i * 3 + 2] as number);
          const q = charges[i] as number;
          real += q * Math.cos(phase);
          imaginary += q * Math.sin(phase);
        }
        // Doubled, for the -k partner this loop does not visit.
        const structure = real * real + imaginary * imaginary;
        energy += 2 * prefactor * weight * structure;

        const { forces } = options;
        if (forces !== undefined) {
          // dE/dr_i = -2 * 2 * prefactor * weight * q_i * k * Im(e^{ik·r_i} conj(S))
          const scale = 4 * prefactor * weight;
          for (let i = 0; i < n; i++) {
            const phase =
              kx * (positions[i * 3] as number) +
              ky * (positions[i * 3 + 1] as number) +
              kz * (positions[i * 3 + 2] as number);
            const q = charges[i] as number;
            const sin = Math.sin(phase);
            const cos = Math.cos(phase);
            // Im(e^{ikr} S*) = sin(phase) * real - cos(phase) * imaginary
            const term = scale * q * (sin * real - cos * imaginary);
            forces[i * 3] = (forces[i * 3] as number) + term * kx;
            forces[i * 3 + 1] = (forces[i * 3 + 1] as number) + term * ky;
            forces[i * 3 + 2] = (forces[i * 3 + 2] as number) + term * kz;
          }
        }
      }
    }
  }
  return { energy, vectors, halfWidths };
}

/** A convenience so a caller does not have to remember the cutoff bound. */
export function ewaldUsableCutoff(box: BoxVectors): number {
  return maximumCutoff(box);
}

export type { BoxVectors, Vec3 };
