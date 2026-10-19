/**
 * Which atoms belong to the same molecule.
 *
 * The barostat needs this and nothing else does. `MonteCarloBarostat` scales
 * the box and then moves each molecule's *centroid* by the same factor,
 * translating every atom in it rigidly -- so internal geometry is preserved
 * exactly and constrained bonds stay constrained. Scaling atoms individually
 * would stretch every bond in the system by the same factor, which is a
 * different and much worse move: the rigid-water constraints would be violated
 * by 1% and the solver would have to undo it every attempt.
 *
 * A molecule is a connected component of the graph whose edges are the bonded
 * interactions and the constraints. Taking all of bonds, angles, torsions,
 * exceptions and constraints as edges cannot over-connect: every one of them is
 * an intramolecular term by construction, so the components are the molecules
 * whichever subset is used. Using all of them is what makes it robust to a
 * system where, say, a rigid water has constraints and no bonds at all -- which
 * is exactly what TIP3P with `rigidWater=True` produces, and what would come
 * out as three separate molecules from bonds alone.
 */

import type { OpenMMSystem } from "./system.js";

/** Union-find with path halving and union by size. */
class DisjointSet {
  private readonly parent: Int32Array;
  private readonly size: Int32Array;

  constructor(count: number) {
    this.parent = new Int32Array(count);
    this.size = new Int32Array(count).fill(1);
    for (let i = 0; i < count; i++) this.parent[i] = i;
  }

  find(x: number): number {
    let node = x;
    while ((this.parent[node] as number) !== node) {
      // Path halving: cheaper than full compression and the same asymptotics.
      this.parent[node] = this.parent[this.parent[node] as number] as number;
      node = this.parent[node] as number;
    }
    return node;
  }

  union(a: number, b: number): void {
    let x = this.find(a);
    let y = this.find(b);
    if (x === y) return;
    if ((this.size[x] as number) < (this.size[y] as number)) {
      const swap = x;
      x = y;
      y = swap;
    }
    this.parent[y] = x;
    this.size[x] = (this.size[x] as number) + (this.size[y] as number);
  }
}

export interface Molecules {
  /** Which molecule each atom belongs to. */
  readonly moleculeOf: Int32Array;
  readonly count: number;
  /** `count + 1` prefix sums into `atoms`. */
  readonly starts: Int32Array;
  /** Atom indices, grouped by molecule and ascending within each. */
  readonly atoms: Int32Array;
  /** The largest molecule's atom count, which is usually the solute's. */
  readonly largest: number;
}

export function findMolecules(system: OpenMMSystem): Molecules {
  const n = system.particleCount;
  const sets = new DisjointSet(n);

  for (let c = 0; c < system.constraintCount; c++) {
    sets.union(system.constraintP1[c] as number, system.constraintP2[c] as number);
  }
  for (const force of system.forces) {
    switch (force.type) {
      case "HarmonicBondForce":
        for (let i = 0; i < force.count; i++) {
          sets.union(force.p1[i] as number, force.p2[i] as number);
        }
        break;
      case "HarmonicAngleForce":
        for (let i = 0; i < force.count; i++) {
          sets.union(force.p1[i] as number, force.p2[i] as number);
          sets.union(force.p2[i] as number, force.p3[i] as number);
        }
        break;
      case "PeriodicTorsionForce":
        for (let i = 0; i < force.count; i++) {
          sets.union(force.p1[i] as number, force.p2[i] as number);
          sets.union(force.p2[i] as number, force.p3[i] as number);
          sets.union(force.p3[i] as number, force.p4[i] as number);
        }
        break;
      case "NonbondedForce":
        for (let e = 0; e < force.exceptionCount; e++) {
          sets.union(force.exceptionP1[e] as number, force.exceptionP2[e] as number);
        }
        break;
      default:
        // CMMotionRemover, MonteCarloBarostat and the restraint carry no
        // bonding information: a restraint ties an atom to a point in space,
        // not to another atom.
        break;
    }
  }

  // Number the molecules by their lowest atom index, so the grouping is a
  // property of the system and not of the order the unions happened in.
  const label = new Int32Array(n).fill(-1);
  const moleculeOf = new Int32Array(n);
  let count = 0;
  for (let atom = 0; atom < n; atom++) {
    const root = sets.find(atom);
    if ((label[root] as number) < 0) {
      label[root] = count;
      count++;
    }
    moleculeOf[atom] = label[root] as number;
  }

  const starts = new Int32Array(count + 1);
  for (let atom = 0; atom < n; atom++) {
    const molecule = moleculeOf[atom] as number;
    starts[molecule + 1] = (starts[molecule + 1] as number) + 1;
  }
  let largest = 0;
  for (let molecule = 0; molecule < count; molecule++) {
    largest = Math.max(largest, starts[molecule + 1] as number);
    starts[molecule + 1] = (starts[molecule + 1] as number) + (starts[molecule] as number);
  }
  const atoms = new Int32Array(n);
  const cursor = Int32Array.from(starts.subarray(0, count));
  for (let atom = 0; atom < n; atom++) {
    const molecule = moleculeOf[atom] as number;
    atoms[cursor[molecule] as number] = atom;
    cursor[molecule] = (cursor[molecule] as number) + 1;
  }

  return { moleculeOf, count, starts, atoms, largest };
}
