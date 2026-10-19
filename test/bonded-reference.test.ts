import { describe, expect, it } from "vitest";
import {
  harmonicAngle,
  harmonicBond,
  periodicTorsion,
} from "../src/reference/forces/bonded.js";
import type {
  HarmonicAngleForce,
  HarmonicBondForce,
  PeriodicTorsionForce,
} from "../src/engine/system.js";
import { fixture, FIXTURE_NAMES, referenceEnergy, type Fixture } from "./fixtures.js";

/**
 * The bonded reference against OpenMM's own numbers, on real systems.
 *
 * Energies come from `facts.json`, which is what OpenMM's double-precision
 * Reference platform reported for this exact `system.xml`. Forces come from
 * `state.xml`, which carries the Reference platform's per-atom forces -- but
 * those are the *total* over every force, so the bonded forces are checked a
 * different way: by the invariants a correct gradient must satisfy, and by a
 * numerical derivative of our own energy.
 */

function findForce<T extends { type: string }>(f: Fixture, type: T["type"]): T {
  const force = f.system.forces.find((x) => x.type === type);
  if (force === undefined) throw new Error(`${f.name} has no ${type}`);
  return force as unknown as T;
}

for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  describe.skipIf(f === null)(`bonded energies for ${name}`, () => {
    it("reproduces OpenMM's HarmonicBondForce energy", () => {
      const force = findForce<HarmonicBondForce>(f!, "HarmonicBondForce");
      const energy = harmonicBond(force, f!.state.positions);
      const want = referenceEnergy(f!, "HarmonicBondForce");
      // Double precision against double precision: the only difference should
      // be the order terms are summed in.
      expect(Math.abs(energy - want) / Math.abs(want)).toBeLessThan(1e-12);
    });

    it("reproduces OpenMM's HarmonicAngleForce energy", () => {
      const force = findForce<HarmonicAngleForce>(f!, "HarmonicAngleForce");
      const energy = harmonicAngle(force, f!.state.positions);
      const want = referenceEnergy(f!, "HarmonicAngleForce");
      expect(Math.abs(energy - want) / Math.abs(want)).toBeLessThan(1e-12);
    });

    it("reproduces OpenMM's PeriodicTorsionForce energy", () => {
      const force = findForce<PeriodicTorsionForce>(f!, "PeriodicTorsionForce");
      const energy = periodicTorsion(force, f!.state.positions);
      const want = referenceEnergy(f!, "PeriodicTorsionForce");
      expect(Math.abs(energy - want) / Math.abs(want)).toBeLessThan(1e-12);
    });
  });

  describe.skipIf(f === null)(`bonded gradients for ${name}`, () => {
    const terms = [
      ["HarmonicBondForce", harmonicBond],
      ["HarmonicAngleForce", harmonicAngle],
      ["PeriodicTorsionForce", periodicTorsion],
    ] as const;

    it.each(terms)("%s forces sum to zero", (type, compute) => {
      const force = findForce(f!, type);
      const forces = new Float64Array(f!.system.particleCount * 3);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      compute(force as any, f!.state.positions, forces);
      let scale = 0;
      for (const value of forces) scale = Math.max(scale, Math.abs(value));
      for (let axis = 0; axis < 3; axis++) {
        let sum = 0;
        for (let i = axis; i < forces.length; i += 3) sum += forces[i] as number;
        // Newton's third law, to the precision the magnitudes allow.
        expect(Math.abs(sum) / scale).toBeLessThan(1e-12);
      }
    });

    it.each(terms)("%s forces match a numerical derivative", (type, compute) => {
      const force = findForce(f!, type);
      const analytic = new Float64Array(f!.system.particleCount * 3);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      compute(force as any, f!.state.positions, analytic);

      // Only atoms this force actually touches, and only a sample of them: a
      // central difference costs two energy evaluations per component.
      const touched = new Set<number>();
      for (const key of ["p1", "p2", "p3", "p4"] as const) {
        const array = (force as unknown as Record<string, Int32Array | undefined>)[key];
        if (array === undefined) continue;
        for (let i = 0; i < array.length; i++) touched.add(array[i] as number);
      }
      const sample = [...touched].slice(0, 12);
      const positions = Float64Array.from(f!.state.positions);
      const h = 1e-6; // nm
      for (const atom of sample) {
        for (let axis = 0; axis < 3; axis++) {
          const index = atom * 3 + axis;
          const original = positions[index] as number;
          positions[index] = original + h;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const plus = compute(force as any, positions);
          positions[index] = original - h;
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const minus = compute(force as any, positions);
          positions[index] = original;
          const numerical = -(plus - minus) / (2 * h);
          const got = analytic[index] as number;
          const magnitude = Math.max(Math.abs(numerical), Math.abs(got), 1);
          expect(Math.abs(got - numerical) / magnitude).toBeLessThan(2e-5);
        }
      }
    });
  });
}
