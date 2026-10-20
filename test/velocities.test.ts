/**
 * Initial velocities, against OpenMM's statistics.
 *
 * The *sample* cannot be compared -- OpenMM draws from SFMT by the Marsaglia
 * polar method and that stream is not reproducible from outside -- so what is
 * compared is everything that does not depend on which deviates came out:
 * the per-component variance, the instantaneous temperature, and whether the
 * constrained relative velocities were projected away.
 *
 * The variance is the interesting one, and it carries two effects at once. A
 * bare Maxwell-Boltzmann draw has `mean(m v^2)/kT = 1` per component. OpenMM
 * measures **0.709** on 1L2Y, because two things happen to it:
 *
 *   * the draw is **shifted by `-(dt/2) F/m`**, which *raises* the variance --
 *     to 1.089 here and 1.419 on 3HTB, where the forces are larger;
 *   * the result is then **projected onto the constraint surface**, which
 *     lowers it, and 2,214 constraints over 2,373 particles removes 31% of the
 *     degrees of freedom.
 *
 * Either one alone gives a plausible wrong answer. Skip the projection and the
 * number goes *up*, toward the 1.0 that looks correct. Skip the shift and it
 * lands near 0.68, which also looks reasonable. Only both together give 0.709,
 * which is why the captured statistics include the unconstrained variance and
 * the forces separately.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { velocitiesAtTemperature } from "../src/reference/velocities.js";
import { splitConstraints } from "../src/engine/constraint-clusters.js";
import { degreesOfFreedom, kineticEnergy } from "../src/reference/kinetic.js";
import { BOLTZMANN_KJ_PER_MOL_K } from "../src/engine/units.js";
import { readNpy } from "./npy.js";
import { fixture, FIXTURE_NAMES } from "./fixtures.js";

interface VelocityFacts {
  readonly temperature_K: number;
  readonly timestep_ps: number;
  readonly kT_kJ_mol: number;
  readonly velocity_constraint_tolerance: number;
  readonly integrators: Record<
    string,
    {
      readonly velocity_time_offset_ps: number;
      readonly temperature_K: { readonly mean: number; readonly min: number; readonly max: number };
      readonly scaled_variance_per_component: { readonly mean: number };
      readonly worst_constrained_relative_velocity_nm_ps: number;
    }
  >;
}

const SEEDS = [1, 2, 3, 4, 5];

for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  const factsPath = `fixtures/${name}/velocity-facts.json`;
  const captured = existsSync(factsPath);

  describe.skipIf(f === null || !captured)(`initial velocities for ${name}`, () => {
    const facts = captured
      ? (JSON.parse(readFileSync(factsPath, "utf8")) as VelocityFacts)
      : undefined;
    const split = f === null ? undefined : splitConstraints(f.system);

    // The forces OpenMM evaluated at these positions, so the shift can be
    // applied without re-evaluating a force field here.
    const forces = captured
      ? readNpy(`fixtures/${name}/velocity-forces.npy`).data
      : new Float64Array(0);
    const offset = facts === undefined ? 0 : facts.timestep_ps / 2;

    const draw = (seed: number, timeOffset = offset): Float64Array =>
      velocitiesAtTemperature(f!.system, {
        temperature: facts!.temperature_K,
        seed,
        positions: f!.state.positions,
        timeOffset,
        forces,
        ...(split === undefined ? {} : { split }),
      });

    it("has the per-component variance OpenMM has, constraints included", () => {
      const kT = facts!.kT_kJ_mol;
      const scaled = SEEDS.map((seed) => {
        const velocities = draw(seed);
        let sum = 0;
        let count = 0;
        for (let i = 0; i < f!.system.particleCount; i++) {
          const mass = f!.system.masses[i] as number;
          if (mass === 0) continue;
          for (let axis = 0; axis < 3; axis++) {
            sum += mass * (velocities[i * 3 + axis] as number) ** 2;
            count++;
          }
        }
        return sum / count / kT;
      });
      const mean = scaled.reduce((a, b) => a + b, 0) / scaled.length;
      const theirs = facts!.integrators["verlet"]!.scaled_variance_per_component.mean;
      // 4% covers the sampling spread over five seeds. The mistakes this
      // catches are far larger: dropping the shift moves 3HTB's figure from
      // 0.936 to 0.68, and dropping the projection moves it to 1.42.
      expect(Math.abs(mean - theirs) / theirs).toBeLessThan(0.04);
    });

    it("lands on OpenMM's instantaneous temperature", () => {
      const dof = degreesOfFreedom(f!.system);
      const temperatures = SEEDS.map(
        (seed) =>
          (2 * kineticEnergy(f!.system, draw(seed))) /
          (dof * BOLTZMANN_KJ_PER_MOL_K),
      );
      const mean = temperatures.reduce((a, b) => a + b, 0) / temperatures.length;
      const theirs = facts!.integrators["verlet"]!.temperature_K;
      // Both means are over five seeds of a quantity whose own spread is a few
      // per cent, so this is a claim about the distribution, not the sample.
      expect(Math.abs(mean - theirs.mean) / theirs.mean).toBeLessThan(0.05);
    });

    it("projects the constrained relative velocities away", () => {
      // The step that makes the variance 0.709 instead of 1. Its absence is
      // invisible in a temperature comparison and obvious here.
      const velocities = draw(1);
      let worst = 0;
      for (let c = 0; c < f!.system.constraintCount; c++) {
        const p1 = (f!.system.constraintP1[c] as number) * 3;
        const p2 = (f!.system.constraintP2[c] as number) * 3;
        let along = 0;
        let length = 0;
        for (let axis = 0; axis < 3; axis++) {
          const d =
            (f!.state.positions[p2 + axis] as number) -
            (f!.state.positions[p1 + axis] as number);
          along +=
            ((velocities[p2 + axis] as number) - (velocities[p1 + axis] as number)) * d;
          length += d * d;
        }
        worst = Math.max(worst, Math.abs(along / Math.sqrt(length)));
      }
      // OpenMM reaches 2.8e-7 nm/ps with the same 1e-5 tolerance; the
      // convergence criterion is on a scaled delta, not on this, so the bound
      // is the same order rather than the same number.
      expect(worst).toBeLessThan(1e-5);
    });

    it("is reproducible from its seed and different between seeds", () => {
      expect(Array.from(draw(7))).toEqual(Array.from(draw(7)));
      expect(Array.from(draw(7))).not.toEqual(Array.from(draw(8)));
    });

    it("leaves frozen particles at rest", () => {
      const frozen = {
        ...f!.system,
        masses: Float64Array.from(f!.system.masses, (mass, i) => (i === 0 ? 0 : mass)),
      };
      const velocities = velocitiesAtTemperature(frozen, {
        temperature: facts!.temperature_K,
        seed: 1,
        positions: f!.state.positions,
        timeOffset: 0,
      });
      expect([velocities[0], velocities[1], velocities[2]]).toEqual([0, 0, 0]);
    });

    it("needs the forces when asked to shift by half a step", () => {
      expect(() =>
        velocitiesAtTemperature(f!.system, {
          temperature: facts!.temperature_K,
          seed: 1,
          positions: f!.state.positions,
          timeOffset: facts!.timestep_ps / 2,
        }),
      ).toThrow(/needs the forces/);
    });

    it("is measurably different without the half-step shift", () => {
      // The shift is not a refinement. On 3HTB it is the difference between a
      // variance of 0.936 and one of 0.68, and between 408 K and 298 K.
      const kT = facts!.kT_kJ_mol;
      const variance = (velocities: Float64Array): number => {
        let sum = 0;
        let count = 0;
        for (let i = 0; i < f!.system.particleCount; i++) {
          const mass = f!.system.masses[i] as number;
          if (mass === 0) continue;
          for (let axis = 0; axis < 3; axis++) {
            sum += mass * (velocities[i * 3 + axis] as number) ** 2;
            count++;
          }
        }
        return sum / count / kT;
      };
      const shifted = variance(draw(3));
      const plain = variance(draw(3, 0));
      expect(shifted).toBeGreaterThan(plain);
      const theirs = facts!.integrators["verlet"]!.scaled_variance_per_component.mean;
      // The shifted one is OpenMM's; the unshifted one is not.
      expect(Math.abs(shifted - theirs)).toBeLessThan(Math.abs(plain - theirs));
    });
  });
}
