/**
 * SETTLE: the analytic solution for a rigid three-atom cluster.
 *
 * Transcribed from `ReferenceSETTLEAlgorithm.cpp`, keeping its variable names so
 * the two can be read side by side. Miyamoto & Kollman, J Comput Chem 13(8)
 * 952-962 (1992) is the method; OpenMM's velocity routine generalises the
 * paper, which silently assumes the two outer masses are equal.
 *
 * Positions: rotate the cluster as a rigid body so all three distances are
 * exact, about its centre of mass, in a frame built from the reference
 * geometry. Analytic, so the result satisfies the constraints to machine
 * precision rather than to a tolerance -- OpenMM's own frames come out at 3e-8
 * relative, far inside its stated 1e-6.
 *
 * Velocities: an exact 3x3 linear solve for the three constraint impulses that
 * make every bond's relative velocity zero.
 *
 * `distance1` is the central atom's two equal bonds and `distance2` the
 * distance between the other two.
 */

import type { OpenMMSystem } from "../engine/system.js";
import type { SettleCluster } from "../engine/constraint-clusters.js";

/**
 * Project `trial` onto the constraint surface for every SETTLE cluster.
 *
 * `reference` are the positions at the start of the step, which define the
 * frame; `trial` is modified in place.
 */
export function settlePositions(
  system: OpenMMSystem,
  clusters: readonly SettleCluster[],
  reference: Float64Array,
  trial: Float64Array,
): void {
  for (const cluster of clusters) {
    const a0 = cluster.central * 3;
    const a1 = cluster.first * 3;
    const a2 = cluster.second * 3;
    const m0 = system.masses[cluster.central] as number;
    const m1 = system.masses[cluster.first] as number;
    const m2 = system.masses[cluster.second] as number;

    // Displacements from the reference positions.
    const xp0 = [
      (trial[a0] as number) - (reference[a0] as number),
      (trial[a0 + 1] as number) - (reference[a0 + 1] as number),
      (trial[a0 + 2] as number) - (reference[a0 + 2] as number),
    ];
    const xp1 = [
      (trial[a1] as number) - (reference[a1] as number),
      (trial[a1 + 1] as number) - (reference[a1 + 1] as number),
      (trial[a1 + 2] as number) - (reference[a1 + 2] as number),
    ];
    const xp2 = [
      (trial[a2] as number) - (reference[a2] as number),
      (trial[a2 + 1] as number) - (reference[a2 + 1] as number),
      (trial[a2 + 2] as number) - (reference[a2 + 2] as number),
    ];

    // The reference bond vectors.
    const xb0 = (reference[a1] as number) - (reference[a0] as number);
    const yb0 = (reference[a1 + 1] as number) - (reference[a0 + 1] as number);
    const zb0 = (reference[a1 + 2] as number) - (reference[a0 + 2] as number);
    const xc0 = (reference[a2] as number) - (reference[a0] as number);
    const yc0 = (reference[a2 + 1] as number) - (reference[a0 + 1] as number);
    const zc0 = (reference[a2 + 2] as number) - (reference[a0 + 2] as number);

    const invTotalMass = 1 / (m0 + m1 + m2);
    const xcom =
      ((xp0[0] as number) * m0 + (xb0 + (xp1[0] as number)) * m1 + (xc0 + (xp2[0] as number)) * m2) *
      invTotalMass;
    const ycom =
      ((xp0[1] as number) * m0 + (yb0 + (xp1[1] as number)) * m1 + (yc0 + (xp2[1] as number)) * m2) *
      invTotalMass;
    const zcom =
      ((xp0[2] as number) * m0 + (zb0 + (xp1[2] as number)) * m1 + (zc0 + (xp2[2] as number)) * m2) *
      invTotalMass;

    const xa1 = (xp0[0] as number) - xcom;
    const ya1 = (xp0[1] as number) - ycom;
    const za1 = (xp0[2] as number) - zcom;
    const xb1 = xb0 + (xp1[0] as number) - xcom;
    const yb1 = yb0 + (xp1[1] as number) - ycom;
    const zb1 = zb0 + (xp1[2] as number) - zcom;
    const xc1 = xc0 + (xp2[0] as number) - xcom;
    const yc1 = yc0 + (xp2[1] as number) - ycom;
    const zc1 = zc0 + (xp2[2] as number) - zcom;

    // An orthonormal frame: z along the reference triangle's normal, x in the
    // plane containing the central atom's displaced position.
    const xaksZd = yb0 * zc0 - zb0 * yc0;
    const yaksZd = zb0 * xc0 - xb0 * zc0;
    const zaksZd = xb0 * yc0 - yb0 * xc0;
    const xaksXd = ya1 * zaksZd - za1 * yaksZd;
    const yaksXd = za1 * xaksZd - xa1 * zaksZd;
    const zaksXd = xa1 * yaksZd - ya1 * xaksZd;
    const xaksYd = yaksZd * zaksXd - zaksZd * yaksXd;
    const yaksYd = zaksZd * xaksXd - xaksZd * zaksXd;
    const zaksYd = xaksZd * yaksXd - yaksZd * xaksXd;

    const axlng = Math.sqrt(xaksXd * xaksXd + yaksXd * yaksXd + zaksXd * zaksXd);
    const aylng = Math.sqrt(xaksYd * xaksYd + yaksYd * yaksYd + zaksYd * zaksYd);
    const azlng = Math.sqrt(xaksZd * xaksZd + yaksZd * yaksZd + zaksZd * zaksZd);
    const trns11 = xaksXd / axlng;
    const trns21 = yaksXd / axlng;
    const trns31 = zaksXd / axlng;
    const trns12 = xaksYd / aylng;
    const trns22 = yaksYd / aylng;
    const trns32 = zaksYd / aylng;
    const trns13 = xaksZd / azlng;
    const trns23 = yaksZd / azlng;
    const trns33 = zaksZd / azlng;

    const xb0d = trns11 * xb0 + trns21 * yb0 + trns31 * zb0;
    const yb0d = trns12 * xb0 + trns22 * yb0 + trns32 * zb0;
    const xc0d = trns11 * xc0 + trns21 * yc0 + trns31 * zc0;
    const yc0d = trns12 * xc0 + trns22 * yc0 + trns32 * zc0;
    const za1d = trns13 * xa1 + trns23 * ya1 + trns33 * za1;
    const xb1d = trns11 * xb1 + trns21 * yb1 + trns31 * zb1;
    const yb1d = trns12 * xb1 + trns22 * yb1 + trns32 * zb1;
    const zb1d = trns13 * xb1 + trns23 * yb1 + trns33 * zb1;
    const xc1d = trns11 * xc1 + trns21 * yc1 + trns31 * zc1;
    const yc1d = trns12 * xc1 + trns22 * yc1 + trns32 * zc1;
    const zc1d = trns13 * xc1 + trns23 * yc1 + trns33 * zc1;

    // Step 2: the canonical rigid triangle in this frame.
    const rc = 0.5 * cluster.distance2;
    let rb = Math.sqrt(cluster.distance1 * cluster.distance1 - rc * rc);
    const ra = rb * (m1 + m2) * invTotalMass;
    rb -= ra;
    const sinphi = za1d / ra;
    const cosphi = Math.sqrt(1 - sinphi * sinphi);
    const sinpsi = (zb1d - zc1d) / (2 * rc * cosphi);
    const cospsi = Math.sqrt(1 - sinpsi * sinpsi);

    const ya2d = ra * cosphi;
    let xb2d = -rc * cospsi;
    const yb2d = -rb * cosphi - rc * sinpsi * sinphi;
    const yc2d = -rb * cosphi + rc * sinpsi * sinphi;
    const xb2d2 = xb2d * xb2d;
    const hh2 =
      4 * xb2d2 + (yb2d - yc2d) * (yb2d - yc2d) + (zb1d - zc1d) * (zb1d - zc1d);
    const deltx =
      2 * xb2d + Math.sqrt(4 * xb2d2 - hh2 + cluster.distance2 * cluster.distance2);
    xb2d -= deltx * 0.5;

    // Step 3: the rotation about z that best matches the trial positions.
    const alpha = xb2d * (xb0d - xc0d) + yb0d * yb2d + yc0d * yc2d;
    const beta = xb2d * (yc0d - yb0d) + xb0d * yb2d + xc0d * yc2d;
    const gamma = xb0d * yb1d - xb1d * yb0d + xc0d * yc1d - xc1d * yc0d;
    const al2be2 = alpha * alpha + beta * beta;
    const sintheta =
      (alpha * gamma - beta * Math.sqrt(al2be2 - gamma * gamma)) / al2be2;

    // Step 4: apply it.
    const costheta = Math.sqrt(1 - sintheta * sintheta);
    const xa3d = -ya2d * sintheta;
    const ya3d = ya2d * costheta;
    const za3d = za1d;
    const xb3d = xb2d * costheta - yb2d * sintheta;
    const yb3d = xb2d * sintheta + yb2d * costheta;
    const zb3d = zb1d;
    const xc3d = -xb2d * costheta - yc2d * sintheta;
    const yc3d = -xb2d * sintheta + yc2d * costheta;
    const zc3d = zc1d;

    // Step 5: back to the laboratory frame.
    const xa3 = trns11 * xa3d + trns12 * ya3d + trns13 * za3d;
    const ya3 = trns21 * xa3d + trns22 * ya3d + trns23 * za3d;
    const za3 = trns31 * xa3d + trns32 * ya3d + trns33 * za3d;
    const xb3 = trns11 * xb3d + trns12 * yb3d + trns13 * zb3d;
    const yb3 = trns21 * xb3d + trns22 * yb3d + trns23 * zb3d;
    const zb3 = trns31 * xb3d + trns32 * yb3d + trns33 * zb3d;
    const xc3 = trns11 * xc3d + trns12 * yc3d + trns13 * zc3d;
    const yc3 = trns21 * xc3d + trns22 * yc3d + trns23 * zc3d;
    const zc3 = trns31 * xc3d + trns32 * yc3d + trns33 * zc3d;

    trial[a0] = (reference[a0] as number) + xcom + xa3;
    trial[a0 + 1] = (reference[a0 + 1] as number) + ycom + ya3;
    trial[a0 + 2] = (reference[a0 + 2] as number) + zcom + za3;
    trial[a1] = (reference[a1] as number) + xcom + xb3 - xb0;
    trial[a1 + 1] = (reference[a1 + 1] as number) + ycom + yb3 - yb0;
    trial[a1 + 2] = (reference[a1 + 2] as number) + zcom + zb3 - zb0;
    trial[a2] = (reference[a2] as number) + xcom + xc3 - xc0;
    trial[a2 + 1] = (reference[a2 + 1] as number) + ycom + yc3 - yc0;
    trial[a2 + 2] = (reference[a2 + 2] as number) + zcom + zc3 - zc0;
  }
}

/**
 * Remove the relative velocity along all three bonds of every SETTLE cluster.
 *
 * An exact solve rather than an iteration: three impulses, one per bond, from a
 * 3x3 system. `velocities` is modified in place.
 */
export function settleVelocities(
  system: OpenMMSystem,
  clusters: readonly SettleCluster[],
  positions: Float64Array,
  velocities: Float64Array,
): void {
  for (const cluster of clusters) {
    const a = cluster.central * 3;
    const b = cluster.first * 3;
    const c = cluster.second * 3;
    const mA = system.masses[cluster.central] as number;
    const mB = system.masses[cluster.first] as number;
    const mC = system.masses[cluster.second] as number;
    const invA = mA === 0 ? 0 : 1 / mA;
    const invB = mB === 0 ? 0 : 1 / mB;
    const invC = mC === 0 ? 0 : 1 / mC;

    const unit = (from: number, to: number): [number, number, number] => {
      const x = (positions[to] as number) - (positions[from] as number);
      const y = (positions[to + 1] as number) - (positions[from + 1] as number);
      const z = (positions[to + 2] as number) - (positions[from + 2] as number);
      const length = Math.sqrt(x * x + y * y + z * z);
      return [x / length, y / length, z / length];
    };
    const eAB = unit(a, b);
    const eBC = unit(b, c);
    const eCA = unit(c, a);

    const relative = (
      from: number,
      to: number,
      e: readonly [number, number, number],
    ): number =>
      ((velocities[to] as number) - (velocities[from] as number)) * e[0] +
      ((velocities[to + 1] as number) - (velocities[from + 1] as number)) * e[1] +
      ((velocities[to + 2] as number) - (velocities[from + 2] as number)) * e[2];
    const vAB = relative(a, b, eAB);
    const vBC = relative(b, c, eBC);
    const vCA = relative(c, a, eCA);

    const dot = (
      p: readonly [number, number, number],
      q: readonly [number, number, number],
    ): number => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    const cA = -dot(eAB, eCA);
    const cB = -dot(eAB, eBC);
    const cC = -dot(eBC, eCA);
    const s2A = 1 - cA * cA;
    const s2B = 1 - cB * cB;
    const s2C = 1 - cC * cC;

    const mABCinv = 1 / (mA * mB * mC);
    const denom =
      (((s2A * mB + s2B * mA) * mC +
        (s2A * mB * mB + 2 * (cA * cB * cC + 1) * mA * mB + s2B * mA * mA)) *
        mC +
        s2C * mA * mB * (mA + mB)) *
      mABCinv;
    const tab =
      ((cB * cC * mA - cA * mB - cA * mC) * vCA +
        (cA * cC * mB - cB * mC - cB * mA) * vBC +
        (s2C * mA * mA * mB * mB * mABCinv + (mA + mB + mC)) * vAB) /
      denom;
    const tbc =
      ((cA * cB * mC - cC * mB - cC * mA) * vCA +
        (s2A * mB * mB * mC * mC * mABCinv + (mA + mB + mC)) * vBC +
        (cA * cC * mB - cB * mA - cB * mC) * vAB) /
      denom;
    const tca =
      ((s2B * mA * mA * mC * mC * mABCinv + (mA + mB + mC)) * vCA +
        (cA * cB * mC - cC * mB - cC * mA) * vBC +
        (cB * cC * mA - cA * mB - cA * mC) * vAB) /
      denom;

    for (let axis = 0; axis < 3; axis++) {
      const ab = eAB[axis] as number;
      const bc = eBC[axis] as number;
      const ca = eCA[axis] as number;
      velocities[a + axis] =
        (velocities[a + axis] as number) + (ab * tab - ca * tca) * invA;
      velocities[b + axis] =
        (velocities[b + axis] as number) + (bc * tbc - ab * tab) * invB;
      velocities[c + axis] =
        (velocities[c + axis] as number) + (ca * tca - bc * tbc) * invC;
    }
  }
}
