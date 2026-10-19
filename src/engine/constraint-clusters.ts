/**
 * Splitting constraints the way OpenMM splits them.
 *
 * OpenMM does not use one constraint algorithm. It finds every closed triangle
 * of three atoms in which each atom holds exactly two constraints and gives
 * those to SETTLE, which solves them analytically; everything else goes to
 * CCMA. That is rigid water, and it is 88 to 90% of the clusters this workload
 * produces.
 *
 * This is a parity requirement rather than a performance choice. SETTLE and
 * SHAKE both satisfy the constraints and land on *different points* of the
 * constraint manifold, about 1.8e-5 nm apart per 2 fs step. See
 * docs/DYNAMICS.md.
 *
 * Transcribed from `ReferenceConstraints::ReferenceConstraints`.
 */

import type { OpenMMSystem } from "./system.js";

/**
 * A three-atom triangle SETTLE can solve: `central` is the atom whose two bonds
 * are equal, and `distance2` is the distance between the other two.
 */
export interface SettleCluster {
  readonly central: number;
  readonly first: number;
  readonly second: number;
  /** The two equal bond lengths, central-first and central-second. */
  readonly distance1: number;
  /** The first-second distance. */
  readonly distance2: number;
}

export interface ConstraintSplit {
  readonly settle: readonly SettleCluster[];
  /** Indices into the system's constraint list, for the iterative solver. */
  readonly iterative: Int32Array;
  /** True for any atom handled by SETTLE. */
  readonly isSettleAtom: Uint8Array;
}

/**
 * Classify every constraint.
 *
 * 🔴 The two distances are compared after rounding to f32, because OpenMM
 * stores them in a `float` before comparing. A pair of bonds differing in the
 * ninth decimal is "equal" to OpenMM and must be here too, or a cluster ends up
 * in a different solver and the trajectory diverges.
 */
export function splitConstraints(system: OpenMMSystem): ConstraintSplit {
  const n = system.particleCount;
  const count = system.constraintCount;

  // How many constraints each atom is in, ignoring any between two frozen
  // particles, as OpenMM does.
  const constraintCount = new Int32Array(n);
  const live = new Uint8Array(count);
  for (let c = 0; c < count; c++) {
    const p1 = system.constraintP1[c] as number;
    const p2 = system.constraintP2[c] as number;
    if ((system.masses[p1] as number) === 0 && (system.masses[p2] as number) === 0) {
      continue;
    }
    live[c] = 1;
    constraintCount[p1] = (constraintCount[p1] as number) + 1;
    constraintCount[p2] = (constraintCount[p2] as number) + 1;
  }

  // Candidate partners: only atoms where both ends hold exactly two
  // constraints can be part of a SETTLE triangle.
  const partners = new Map<number, Map<number, number>>();
  const link = (a: number, b: number, distance: number): void => {
    let map = partners.get(a);
    if (map === undefined) {
      map = new Map();
      partners.set(a, map);
    }
    map.set(b, Math.fround(distance));
  };
  for (let c = 0; c < count; c++) {
    if (live[c] === 0) continue;
    const p1 = system.constraintP1[c] as number;
    const p2 = system.constraintP2[c] as number;
    if ((constraintCount[p1] as number) === 2 && (constraintCount[p2] as number) === 2) {
      const distance = system.constraintDistance[c] as number;
      link(p1, p2, distance);
      link(p2, p1, distance);
    }
  }

  // Keep only the ones that close into a triangle, and name each triangle once,
  // by its lowest-numbered atom.
  const roots: number[] = [];
  for (const [atom, map] of partners) {
    if (map.size !== 2) {
      map.clear();
      continue;
    }
    const [first, second] = [...map.keys()] as [number, number];
    const firstPartners = partners.get(first);
    const secondPartners = partners.get(second);
    if (
      firstPartners === undefined ||
      secondPartners === undefined ||
      firstPartners.size !== 2 ||
      secondPartners.size !== 2 ||
      !firstPartners.has(second)
    ) {
      map.clear();
      continue;
    }
    if (atom < first && atom < second) roots.push(atom);
  }

  const settle: SettleCluster[] = [];
  const isSettleAtom = new Uint8Array(n);
  for (const p1 of roots) {
    const map = partners.get(p1);
    if (map === undefined || map.size !== 2) continue;
    const [p2, p3] = [...map.keys()] as [number, number];
    const d12 = map.get(p2) as number;
    const d13 = map.get(p3) as number;
    const d23 = partners.get(p2)?.get(p3);
    if (d23 === undefined) continue;

    let cluster: SettleCluster | undefined;
    if (d12 === d13) {
      cluster = { central: p1, first: p2, second: p3, distance1: d12, distance2: d23 };
    } else if (d12 === d23) {
      cluster = { central: p2, first: p1, second: p3, distance1: d12, distance2: d13 };
    } else if (d13 === d23) {
      cluster = { central: p3, first: p1, second: p2, distance1: d13, distance2: d12 };
    }
    // No two sides equal: SETTLE's geometry assumes an isoceles triangle, so
    // this one goes to the iterative solver instead.
    if (cluster === undefined) continue;
    settle.push(cluster);
    isSettleAtom[p1] = 1;
    isSettleAtom[p2] = 1;
    isSettleAtom[p3] = 1;
  }

  const iterative: number[] = [];
  for (let c = 0; c < count; c++) {
    if (live[c] === 0) continue;
    const p1 = system.constraintP1[c] as number;
    if (isSettleAtom[p1] === 1) continue;
    iterative.push(c);
  }
  return { settle, iterative: Int32Array.from(iterative), isSettleAtom };
}
