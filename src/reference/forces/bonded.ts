/**
 * Bonded forces in double precision, as OpenMM defines them.
 *
 * The independent implementation the WGSL kernels are differentially tested
 * against.  Written for clarity, never for speed, and it never imports from
 * `src/gpu/`.
 *
 * Each function returns the energy in kJ/mol and, when given a `forces` array,
 * accumulates -dE/dr into it in kJ/mol/nm.  Accumulating rather than assigning
 * is what the engine needs, and it lets a caller sum several force terms into
 * one array the way a `System` does.
 *
 * None of these applies periodic boundary conditions.  Both committed fixtures
 * carry `usesPeriodic="0"` on all three bonded forces, which is what
 * `ForceField.createSystem` produces: a bonded term spans at most four
 * covalently connected atoms, so it cannot straddle a box boundary unless the
 * molecule has already been wrapped, and OpenMM does not wrap.  The flag is
 * parsed and asserted rather than assumed.
 */

import type {
  HarmonicAngleForce,
  HarmonicBondForce,
  PeriodicTorsionForce,
} from "../../engine/system.js";

export class PeriodicBondedForce extends Error {
  constructor(type: string) {
    super(
      `${type} has usesPeriodic set. The reference implementation does not ` +
        "apply periodic boundary conditions to bonded terms, because nothing " +
        "this engine builds produces them. See src/reference/forces/bonded.ts.",
    );
    this.name = "PeriodicBondedForce";
  }
}

/** E = (1/2) k (r - r0)^2 */
export function harmonicBond(
  force: HarmonicBondForce,
  positions: Float64Array,
  forces?: Float64Array,
): number {
  if (force.usesPeriodic) throw new PeriodicBondedForce(force.type);
  let energy = 0;
  for (let i = 0; i < force.count; i++) {
    const a = (force.p1[i] as number) * 3;
    const b = (force.p2[i] as number) * 3;
    const dx = (positions[b] as number) - (positions[a] as number);
    const dy = (positions[b + 1] as number) - (positions[a + 1] as number);
    const dz = (positions[b + 2] as number) - (positions[a + 2] as number);
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz);
    const dr = r - (force.length[i] as number);
    const k = force.k[i] as number;
    energy += 0.5 * k * dr * dr;
    if (forces !== undefined) {
      // F on p1 points along (p2 - p1) when the bond is stretched.
      const scale = (k * dr) / r;
      forces[a] = (forces[a] as number) + scale * dx;
      forces[a + 1] = (forces[a + 1] as number) + scale * dy;
      forces[a + 2] = (forces[a + 2] as number) + scale * dz;
      forces[b] = (forces[b] as number) - scale * dx;
      forces[b + 1] = (forces[b + 1] as number) - scale * dy;
      forces[b + 2] = (forces[b + 2] as number) - scale * dz;
    }
  }
  return energy;
}

/** E = (1/2) k (theta - theta0)^2 */
export function harmonicAngle(
  force: HarmonicAngleForce,
  positions: Float64Array,
  forces?: Float64Array,
): number {
  if (force.usesPeriodic) throw new PeriodicBondedForce(force.type);
  let energy = 0;
  for (let i = 0; i < force.count; i++) {
    const a = (force.p1[i] as number) * 3;
    const b = (force.p2[i] as number) * 3;
    const c = (force.p3[i] as number) * 3;
    // The two bond vectors, both pointing away from the central atom.
    const v1x = (positions[a] as number) - (positions[b] as number);
    const v1y = (positions[a + 1] as number) - (positions[b + 1] as number);
    const v1z = (positions[a + 2] as number) - (positions[b + 2] as number);
    const v2x = (positions[c] as number) - (positions[b] as number);
    const v2y = (positions[c + 1] as number) - (positions[b + 1] as number);
    const v2z = (positions[c + 2] as number) - (positions[b + 2] as number);

    const r1sq = v1x * v1x + v1y * v1y + v1z * v1z;
    const r2sq = v2x * v2x + v2y * v2y + v2z * v2z;
    const dot = v1x * v2x + v1y * v2y + v1z * v2z;
    const cosine = Math.min(1, Math.max(-1, dot / Math.sqrt(r1sq * r2sq)));
    const theta = Math.acos(cosine);
    const k = force.k[i] as number;
    const dTheta = theta - (force.angle[i] as number);
    energy += 0.5 * k * dTheta * dTheta;
    if (forces === undefined) continue;

    // n = v1 x v2 is perpendicular to the angle's plane, and |n| = r1 r2 sin(theta).
    const nx = v1y * v2z - v1z * v2y;
    const ny = v1z * v2x - v1x * v2z;
    const nz = v1x * v2y - v1y * v2x;
    // A collinear angle has no well-defined plane and no restoring direction.
    // OpenMM floors the same quantity at 1e-6 rather than dividing by zero.
    const nNorm = Math.max(Math.sqrt(nx * nx + ny * ny + nz * nz), 1e-6);
    const dEdTheta = k * dTheta;
    // dTheta/dp1 = (v1 x n) / (r1^2 |n|), dTheta/dp3 = -(v2 x n) / (r2^2 |n|)
    const termA = dEdTheta / (r1sq * nNorm);
    const termC = -dEdTheta / (r2sq * nNorm);
    const f1x = -termA * (v1y * nz - v1z * ny);
    const f1y = -termA * (v1z * nx - v1x * nz);
    const f1z = -termA * (v1x * ny - v1y * nx);
    const f3x = -termC * (v2y * nz - v2z * ny);
    const f3y = -termC * (v2z * nx - v2x * nz);
    const f3z = -termC * (v2x * ny - v2y * nx);
    forces[a] = (forces[a] as number) + f1x;
    forces[a + 1] = (forces[a + 1] as number) + f1y;
    forces[a + 2] = (forces[a + 2] as number) + f1z;
    forces[c] = (forces[c] as number) + f3x;
    forces[c + 1] = (forces[c + 1] as number) + f3y;
    forces[c + 2] = (forces[c + 2] as number) + f3z;
    // The central atom takes whatever keeps the triple force-free, which also
    // keeps the torque zero.
    forces[b] = (forces[b] as number) - f1x - f3x;
    forces[b + 1] = (forces[b + 1] as number) - f1y - f3y;
    forces[b + 2] = (forces[b + 2] as number) - f1z - f3z;
  }
  return energy;
}

/** E = k (1 + cos(periodicity * phi - phase)) */
export function periodicTorsion(
  force: PeriodicTorsionForce,
  positions: Float64Array,
  forces?: Float64Array,
): number {
  if (force.usesPeriodic) throw new PeriodicBondedForce(force.type);
  let energy = 0;
  for (let i = 0; i < force.count; i++) {
    const a = (force.p1[i] as number) * 3;
    const b = (force.p2[i] as number) * 3;
    const c = (force.p3[i] as number) * 3;
    const d = (force.p4[i] as number) * 3;

    const b1x = (positions[b] as number) - (positions[a] as number);
    const b1y = (positions[b + 1] as number) - (positions[a + 1] as number);
    const b1z = (positions[b + 2] as number) - (positions[a + 2] as number);
    const b2x = (positions[c] as number) - (positions[b] as number);
    const b2y = (positions[c + 1] as number) - (positions[b + 1] as number);
    const b2z = (positions[c + 2] as number) - (positions[b + 2] as number);
    const b3x = (positions[d] as number) - (positions[c] as number);
    const b3y = (positions[d + 1] as number) - (positions[c + 1] as number);
    const b3z = (positions[d + 2] as number) - (positions[c + 2] as number);

    // Normals of the two planes the torsion is measured between.
    const n1x = b1y * b2z - b1z * b2y;
    const n1y = b1z * b2x - b1x * b2z;
    const n1z = b1x * b2y - b1y * b2x;
    const n2x = b2y * b3z - b2z * b3y;
    const n2y = b2z * b3x - b2x * b3z;
    const n2z = b2x * b3y - b2y * b3x;

    const n1sq = n1x * n1x + n1y * n1y + n1z * n1z;
    const n2sq = n2x * n2x + n2y * n2y + n2z * n2z;
    const b2Norm = Math.sqrt(b2x * b2x + b2y * b2y + b2z * b2z);

    // phi via atan2 rather than acos: it is defined over the whole circle and
    // does not lose precision where the planes are nearly parallel.
    const x = n1x * n2x + n1y * n2y + n1z * n2z;
    const y = b2Norm * (b1x * n2x + b1y * n2y + b1z * n2z);
    const phi = Math.atan2(y, x);

    const n = force.periodicity[i] as number;
    const phase = force.phase[i] as number;
    const k = force.k[i] as number;
    energy += k * (1 + Math.cos(n * phi - phase));
    if (forces === undefined) continue;

    // dE/dphi, then the chain rule onto the four atoms.
    const dEdPhi = -k * n * Math.sin(n * phi - phase);
    if (n1sq === 0 || n2sq === 0) continue; // degenerate geometry

    // The two gradients that have a closed form, for the phi convention above:
    //   dphi/dp1 = -|b2| n1 / |n1|^2      dphi/dp4 = +|b2| n2 / |n2|^2
    // Both were confirmed against a central difference of `phi` itself rather
    // than taken from a textbook, because the sign depends on which way b1, b2
    // and b3 are defined and on the argument order inside atan2.
    const f1Scale = (dEdPhi * b2Norm) / n1sq;
    const f4Scale = (-dEdPhi * b2Norm) / n2sq;
    const f1x = f1Scale * n1x;
    const f1y = f1Scale * n1y;
    const f1z = f1Scale * n1z;
    const f4x = f4Scale * n2x;
    const f4y = f4Scale * n2y;
    const f4z = f4Scale * n2z;

    // The inner two atoms take the rest. How far along b2 each end's lever arm
    // projects decides how it is split:
    //   dphi/dp2 = -(1 + s) dphi/dp1 + t dphi/dp4
    //   dphi/dp3 =       s  dphi/dp1 - (1 + t) dphi/dp4
    // which also makes the four forces sum to zero, as the test asserts.
    const b2sq = b2Norm * b2Norm;
    const s = (b1x * b2x + b1y * b2y + b1z * b2z) / b2sq;
    const t = (b3x * b2x + b3y * b2y + b3z * b2z) / b2sq;
    const f2x = -(1 + s) * f1x + t * f4x;
    const f2y = -(1 + s) * f1y + t * f4y;
    const f2z = -(1 + s) * f1z + t * f4z;
    const f3x = s * f1x - (1 + t) * f4x;
    const f3y = s * f1y - (1 + t) * f4y;
    const f3z = s * f1z - (1 + t) * f4z;

    forces[a] = (forces[a] as number) + f1x;
    forces[a + 1] = (forces[a + 1] as number) + f1y;
    forces[a + 2] = (forces[a + 2] as number) + f1z;
    forces[b] = (forces[b] as number) + f2x;
    forces[b + 1] = (forces[b + 1] as number) + f2y;
    forces[b + 2] = (forces[b + 2] as number) + f2z;
    forces[c] = (forces[c] as number) + f3x;
    forces[c + 1] = (forces[c + 1] as number) + f3y;
    forces[c + 2] = (forces[c + 2] as number) + f3z;
    forces[d] = (forces[d] as number) + f4x;
    forces[d + 1] = (forces[d + 1] as number) + f4y;
    forces[d + 2] = (forces[d + 2] as number) + f4z;
  }
  return energy;
}
