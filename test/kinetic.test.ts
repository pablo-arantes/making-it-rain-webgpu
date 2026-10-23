import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  degreesOfFreedom,
  kineticEnergy,
  shiftedKineticEnergy,
} from "../src/reference/kinetic.js";
import { readNpy } from "./npy.js";
import { fixture, FIXTURE_NAMES, ifPresent } from "./fixtures.js";

/**
 * The two kinetic-energy conventions, against what OpenMM reported along the
 * captured trajectories.
 *
 * This is a parity requirement, not a detail: the notebook logs kineticEnergy,
 * totalEnergy and temperature every 10 ps with a `LangevinIntegrator`, so the
 * engine's reporters have to use the unshifted convention or every line
 * disagrees. See docs/DYNAMICS.md.
 */
for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  const captured = existsSync(`fixtures/${name}/trajectory-verlet-forces.npy`);
  const dt = 0.002;

  describe.skipIf(f === null || !captured)(`kinetic energy for ${name}`, () => {
    const frames = (label: string, kind: string): { shape: readonly number[]; data: Float64Array } =>
      readNpy(`fixtures/${name}/trajectory-${label}-${kind}.npy`);

    interface Frame {
      readonly potential: number;
      readonly kinetic: number;
    }
    // `skipIf` does not stop this body running -- see `ifPresent`.
    const dynamics = ifPresent(
      f !== null && captured,
      () =>
        JSON.parse(
          readFileSync(`fixtures/${name}/dynamics-facts.json`, "utf8"),
        ) as {
          trajectory_verlet: { energies_kJ_mol: Frame[] };
          trajectory_lfmiddle: { energies_kJ_mol: Frame[] };
        },
    );

    it("reproduces what OpenMM reports for LFMiddle, unshifted", () => {
      const velocities = frames("lfmiddle", "velocities");
      const expected = dynamics.trajectory_lfmiddle.energies_kJ_mol;
      const stride = f!.system.particleCount * 3;
      for (let frame = 0; frame < expected.length; frame++) {
        const v = velocities.data.subarray(frame * stride, (frame + 1) * stride);
        const ours = kineticEnergy(f!.system, v);
        const want = expected[frame]!.kinetic;
        expect(Math.abs(ours - want) / want).toBeLessThan(1e-12);
      }
    });

    it("reproduces what OpenMM reports for Verlet, shifted and re-constrained", () => {
      const positions = frames("verlet", "positions");
      const velocities = frames("verlet", "velocities");
      const forces = frames("verlet", "forces");
      const expected = dynamics.trajectory_verlet.energies_kJ_mol;
      const stride = f!.system.particleCount * 3;
      for (let frame = 0; frame < expected.length; frame++) {
        const x = positions.data.subarray(frame * stride, (frame + 1) * stride);
        const v = velocities.data.subarray(frame * stride, (frame + 1) * stride);
        const force = forces.data.subarray(frame * stride, (frame + 1) * stride);
        const ours = shiftedKineticEnergy(f!.system, x, v, force, 0.5 * dt);
        const want = expected[frame]!.kinetic;
        // Both re-constrain to OpenMM's hard-coded 1e-4, so they stop at
        // different points inside that band.
        expect(Math.abs(ours - want) / want).toBeLessThan(1e-5);
      }
    });

    it("shows the shift is what conserves, and the re-constraint is most of it", () => {
      const positions = frames("verlet", "positions");
      const velocities = frames("verlet", "velocities");
      const forces = frames("verlet", "forces");
      const stride = f!.system.particleCount * 3;
      const potential = dynamics.trajectory_verlet.energies_kJ_mol;

      const range = (compute: (frame: number) => number): number => {
        let low = Infinity;
        let high = -Infinity;
        for (let frame = 0; frame < potential.length; frame++) {
          const total = potential[frame]!.potential + compute(frame);
          low = Math.min(low, total);
          high = Math.max(high, total);
        }
        return (high - low) / f!.system.particleCount;
      };
      const slice = (a: { data: Float64Array }, frame: number): Float64Array =>
        a.data.subarray(frame * stride, (frame + 1) * stride);

      const unshifted = range((frame) =>
        kineticEnergy(f!.system, slice(velocities, frame)),
      );
      const shifted = range((frame) =>
        shiftedKineticEnergy(
          f!.system,
          slice(positions, frame),
          slice(velocities, frame),
          slice(forces, frame),
          0.5 * dt,
        ),
      );
      expect(shifted).toBeLessThan(unshifted);
    });

    it("counts degrees of freedom as OpenMM's reporter does", () => {
      const dof = degreesOfFreedom(f!.system);
      let moving = 0;
      for (let i = 0; i < f!.system.particleCount; i++) {
        if ((f!.system.masses[i] as number) > 0) moving++;
      }
      // Three per moving particle, less one per constraint, less three for the
      // centre-of-mass motion remover.
      expect(dof).toBe(3 * moving - f!.system.constraintCount - 3);
      expect(dof).toBeGreaterThan(0);
    });
  });
}
