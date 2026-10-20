/**
 * Cross-fed energies: the Phase 3 gate the structural comparison cannot replace.
 *
 * Every force in a prepared `System` matches OpenMM's element for element on
 * the parity fixtures. That proves the two engines assign the same parameters
 * to the same terms. It does not prove they agree on an **energy** — a wrong
 * unit in `serialiseSystem`, or a force written with a convention OpenMM reads
 * differently, survives every structural test and appears only here.
 *
 * So: prepare a system here (`tools/export_prepared.ts`), serialise it, have
 * OpenMM read that file and evaluate it on the double-precision Reference
 * platform (`tools/cross_feed_energy.py`), and evaluate the same system with
 * this project's own reference implementation. Two independent force fields
 * reading one file.
 *
 * The system is deliberately the one with **no OpenMM counterpart**: 1L2Y with
 * hydrogens this code placed and a water box this code filled. OpenMM never
 * built it, so it cannot be compared structurally, and this is the only check
 * available for it.
 *
 * Three details that make the comparison fair rather than approximate:
 *
 * **One force group per force.** A total is a sum, and a bonded term too high
 * cancels against a nonbonded term too low. The tool puts each force in its
 * own group so a disagreement names the force.
 *
 * **The PME mesh is OpenMM's, asked for rather than recomputed.** The Reference
 * platform takes the exact minimum grid; the engine rounds up to a smooth FFT
 * size. Comparing those is comparing two different meshes, and the difference
 * is far larger than round-off.
 *
 * **No constraints.** A constrained bond is *removed* from the
 * HarmonicBondForce, so a constrained system and an unconstrained one have
 * genuinely different bonded energies. The exported system has none; the engine
 * applies them at run time.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseState } from "../src/engine/state.js";
import { parseSystem } from "../src/engine/system.js";
import { evaluatePotential } from "../src/reference/potential.js";
import type { Vec3 } from "../src/engine/system.js";

interface Energies {
  readonly platform: string;
  readonly particles: number;
  readonly constraints: number;
  readonly byForce: Readonly<Record<string, number>>;
  readonly pme: { readonly alpha_per_nm: number; readonly grid: [number, number, number] };
  readonly total: number;
  readonly forceSample: readonly (readonly number[])[];
}

const systemPath = "fixtures/crossfeed/system.xml";
const statePath = "fixtures/crossfeed/state.xml";
const energiesPath = "fixtures/crossfeed/energies.json";
const present =
  existsSync(systemPath) && existsSync(statePath) && existsSync(energiesPath);

/** The sampled atoms `cross_feed_energy.py` records forces for. */
const SAMPLED = [0, 1, 2, 100, 1000] as const;

describe.skipIf(!present)("cross-fed energies for a browser-prepared system", () => {
  const system = present ? parseSystem(readFileSync(systemPath, "utf8")) : undefined;
  const state = present ? parseState(readFileSync(statePath, "utf8")) : undefined;
  const reference = present
    ? (JSON.parse(readFileSync(energiesPath, "utf8")) as Energies)
    : undefined;
  const evaluated =
    present && system !== undefined && state !== undefined && reference !== undefined
      ? evaluatePotential(system, state.positions, state.boxVectors, {
          grid: reference.pme.grid as unknown as Vec3,
        })
      : undefined;

  /** Our nonbonded is reported as direct + reciprocal; OpenMM has one group. */
  const ours = (): Record<string, number> => {
    const merged: Record<string, number> = {};
    for (const [name, value] of Object.entries(evaluated!.byForce)) {
      const key = name.startsWith("NonbondedForce") ? "NonbondedForce" : name;
      merged[key] = (merged[key] ?? 0) + value;
    }
    return merged;
  };
  const theirs = (): Record<string, number> => {
    const out: Record<string, number> = {};
    // Keys are "<group>:<ForceName>".
    for (const [key, value] of Object.entries(reference!.byForce)) {
      out[key.slice(key.indexOf(":") + 1)] = value;
    }
    return out;
  };

  it("was evaluated by OpenMM at double precision, with no constraints", () => {
    // If this is not the Reference platform the tolerances below are wrong:
    // a mixed-precision platform disagrees with a double-precision reference
    // at about 1e-5 relative, not 1e-12.
    expect(reference!.platform).toBe("Reference");
    expect(reference!.constraints).toBe(0);
    expect(reference!.particles).toBe(system!.particleCount);
  });

  it("reads the same system OpenMM read", () => {
    // OpenMM parsed the file this project wrote, which is itself the first
    // result: a serialiser that produced something OpenMM refuses would fail
    // in the tool rather than here, and a file it *misreads* would show up as
    // an energy disagreement below.
    expect(system!.particleCount).toBe(2400);
    expect(system!.forces.map((f) => f.type).sort()).toEqual([
      "HarmonicAngleForce",
      "HarmonicBondForce",
      "NonbondedForce",
      "PeriodicTorsionForce",
    ]);
  });

  it("agrees on every bonded force group to double precision", () => {
    const mine = ours();
    const openmm = theirs();
    for (const name of [
      "HarmonicBondForce",
      "HarmonicAngleForce",
      "PeriodicTorsionForce",
    ]) {
      const want = openmm[name] as number;
      const got = mine[name] as number;
      expect(got, `${name} missing`).toBeDefined();
      // Two double-precision sums over the same terms in a different order.
      expect(Math.abs(got - want) / Math.max(1, Math.abs(want))).toBeLessThan(1e-13);
    }
  });

  it("agrees on the nonbonded group, direct and reciprocal together", () => {
    const want = theirs()["NonbondedForce"] as number;
    const got = ours()["NonbondedForce"] as number;
    // Looser than the bonded terms, and legitimately so: this is a sum of a
    // few million direct-space pairs against a 25^3 mesh, where the two
    // engines differ in summation order and in the FFT they use. 1e-10 is
    // still far below any physical significance -- the term is -25,005 kJ/mol,
    // so this is a fraction of a micro-kJ.
    expect(Math.abs(got - want) / Math.abs(want)).toBeLessThan(1e-10);
  });

  it("agrees on the total potential energy", () => {
    expect(
      Math.abs(evaluated!.energy - reference!.total) / Math.abs(reference!.total),
    ).toBeLessThan(1e-10);
  });

  it("agrees on the forces, not only the energy", () => {
    // 🔴 An energy can be right while a gradient is wrong — a sign convention
    // in one term, or a derivative that happens not to affect the value at
    // this geometry. The forces are what the integrator uses, so they are what
    // has to agree.
    let worst = 0;
    for (const [slot, atom] of SAMPLED.entries()) {
      const want = reference!.forceSample[slot] as readonly number[];
      for (let axis = 0; axis < 3; axis++) {
        worst = Math.max(
          worst,
          Math.abs((evaluated!.forces[atom * 3 + axis] as number) - (want[axis] as number)),
        );
      }
    }
    // kJ/mol/nm, against components of order 1e2-1e3.
    expect(worst).toBeLessThan(1e-6);
  });
});
