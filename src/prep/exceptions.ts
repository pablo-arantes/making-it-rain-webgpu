/**
 * Nonbonded exceptions: what Amber calls 1-2, 1-3 and 1-4.
 *
 * Atoms close along the bond graph do not interact through the plain
 * Lennard-Jones and Coulomb terms, because the bonded terms already describe
 * them. OpenMM expresses that as *exceptions* to the nonbonded force, created
 * by `NonbondedForce::createExceptionsFromBonds`, which this transcribes.
 *
 * - **1-2 and 1-3** are removed outright: `chargeProd = 0`, `epsilon = 0`.
 * - **1-4** is kept but scaled, by `coulomb14scale` (5/6 for Amber) and
 *   `lj14scale` (1/2), with sigma the arithmetic mean and epsilon the geometric
 *   mean of the pair.
 *
 * 🔴 A removed exception gets `sigma = 1.0`, not the pair's mean. It looks like
 * a placeholder and it is -- epsilon is zero, so sigma cannot affect the energy
 * -- but it is in the file and in the `System` XML, so a comparison that
 * computed the mean instead would differ on 875 of 1L2Y's 1687 exceptions
 * while being physically identical. Matching the serialised form is what lets
 * the notebook stay the oracle.
 *
 * The 1-4 test is "exactly three bonds away": not 1-2, and not within two
 * bonds. Testing only "not 1-2 and not 1-3" would scale a 1-5 pair in any ring
 * where a shorter path also exists, which is most of a protein's aromatics.
 */

import type { AssignedAtom } from "./assign.js";
import type { Pair } from "./bonded.js";

export interface Exception {
  readonly atoms: Pair;
  /** e^2, already scaled. */
  readonly chargeProd: number;
  /** nm. */
  readonly sigma: number;
  /** kJ/mol, already scaled. */
  readonly epsilon: number;
}

/**
 * Pairs within `maxSeparation` bonds, as sets per atom.
 *
 * Breadth-first from each atom rather than OpenMM's two-pass set union, which
 * is the same relation by a clearer route. The pass structure there is an
 * artefact of avoiding a queue, not a different definition -- and this is
 * checked against the serialised exceptions, so an off-by-one in the
 * separation would show up as thousands of differing pairs rather than as a
 * subtle shift.
 */
export function withinBonds(
  atomCount: number,
  bonds: readonly Pair[],
  maxSeparation: number,
): readonly ReadonlySet<number>[] {
  const bonded: Set<number>[] = Array.from({ length: atomCount }, () => new Set());
  for (const [a, b] of bonds) {
    (bonded[a] as Set<number>).add(b);
    (bonded[b] as Set<number>).add(a);
  }

  const within: Set<number>[] = Array.from({ length: atomCount }, () => new Set());
  for (let start = 0; start < atomCount; start++) {
    const distance = new Map<number, number>([[start, 0]]);
    let frontier = [start];
    for (let step = 1; step <= maxSeparation; step++) {
      const next: number[] = [];
      for (const atom of frontier) {
        for (const neighbour of bonded[atom] as Set<number>) {
          if (distance.has(neighbour)) continue;
          distance.set(neighbour, step);
          next.push(neighbour);
          (within[start] as Set<number>).add(neighbour);
        }
      }
      frontier = next;
    }
  }
  return within;
}

/**
 * Every exception, in OpenMM's own order.
 *
 * It walks `i` ascending and emits each partner `j < i` as the pair `(j, i)`,
 * so the list is ordered by the *second* atom and every pair is ascending.
 * Measured against the built `System` rather than assumed: 1L2Y's 1687
 * exceptions are sorted by `(second, first)` and not by `(first, second)`.
 * Worth reproducing, because that is the order they take in a serialised
 * `System`, and the engine contract is that XML.
 */
export function findExceptions(
  atoms: readonly AssignedAtom[],
  bonds: readonly Pair[],
  coulomb14Scale: number,
  lj14Scale: number,
): readonly Exception[] {
  const within3 = withinBonds(atoms.length, bonds, 3);
  const within2 = withinBonds(atoms.length, bonds, 2);
  const bonded12 = withinBonds(atoms.length, bonds, 1);

  const exceptions: Exception[] = [];
  for (let i = 0; i < atoms.length; i++) {
    const partners = [...(within3[i] as ReadonlySet<number>)]
      .filter((j) => j < i)
      .sort((a, b) => a - b);
    for (const j of partners) {
      const isOneFour =
        !(bonded12[i] as ReadonlySet<number>).has(j) &&
        !(within2[i] as ReadonlySet<number>).has(j);
      if (isOneFour) {
        const first = atoms[j] as AssignedAtom;
        const second = atoms[i] as AssignedAtom;
        exceptions.push({
          atoms: [j, i],
          chargeProd: coulomb14Scale * first.charge * second.charge,
          sigma: 0.5 * (first.sigma + second.sigma),
          epsilon: lj14Scale * Math.sqrt(first.epsilon * second.epsilon),
        });
      } else {
        // 🔴 sigma 1.0, matching the serialised form. See the module comment.
        exceptions.push({ atoms: [j, i], chargeProd: 0, sigma: 1, epsilon: 0 });
      }
    }
  }
  return exceptions;
}
