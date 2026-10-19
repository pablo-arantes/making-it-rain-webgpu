import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  langevinMiddleStep,
  verletStep,
  type IntegratorState,
} from "../src/reference/integrator.js";
import { readNpy } from "./npy.js";
import { fixture, FIXTURE_NAMES } from "./fixtures.js";

/**
 * Both integrators against OpenMM's own trajectories, step by step.
 *
 * Driven by the **forces OpenMM recorded at each frame**, not by our own force
 * evaluation. That isolates the integrator and the constraint solver: a
 * disagreement here cannot be a force-field error leaking in, which is the one
 * thing that would make this test hard to read.
 *
 * Both trajectories were captured with the thermostat off -- `VerletIntegrator`
 * has none, and `LangevinIntegrator` with friction 0 has `noisescale = 0` -- so
 * both are deterministic without reproducing OpenMM's random stream.
 *
 * What limits the agreement is the constraint solver, not the integrator: we
 * use SHAKE where OpenMM uses SETTLE for rigid water. See the note on the
 * position bound below.
 */
for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  const captured = existsSync(`fixtures/${name}/trajectory-verlet-forces.npy`);
  const TIME_STEP = 0.002;
  const TOLERANCE = 1e-6;

  describe.skipIf(f === null || !captured)(`integrators for ${name}`, () => {
    const stride = (f?.system.particleCount ?? 0) * 3;
    const load = (label: string, kind: string): Float64Array =>
      readNpy(`fixtures/${name}/trajectory-${label}-${kind}.npy`).data;
    const frame = (array: Float64Array, index: number): Float64Array =>
      array.subarray(index * stride, (index + 1) * stride);
    const steps = (
      JSON.parse(
        readFileSync(`fixtures/${name}/dynamics-facts.json`, "utf8"),
      ) as { trajectory_verlet: { steps: number } }
    ).trajectory_verlet.steps;

    /** The worst per-component difference between two arrays. */
    const worst = (a: Float64Array, b: Float64Array): number => {
      let value = 0;
      for (let i = 0; i < a.length; i++) {
        value = Math.max(value, Math.abs((a[i] as number) - (b[i] as number)));
      }
      return value;
    };

    it("reproduces OpenMM's Verlet step from each frame", () => {
      const positions = load("verlet", "positions");
      const velocities = load("verlet", "velocities");
      const forces = load("verlet", "forces");
      let worstPosition = 0;
      let worstVelocity = 0;
      for (let step = 0; step < steps; step++) {
        // Start each step from OpenMM's own frame, so errors cannot accumulate
        // and every step is tested independently.
        const state: IntegratorState = {
          positions: Float64Array.from(frame(positions, step)),
          velocities: Float64Array.from(frame(velocities, step)),
        };
        const result = verletStep(
          f!.system,
          state,
          frame(forces, step),
          TIME_STEP,
          TOLERANCE,
        );
        expect(result.positionConstraints.converged).toBe(true);
        worstPosition = Math.max(
          worstPosition,
          worst(state.positions, frame(positions, step + 1)),
        );
        worstVelocity = Math.max(
          worstVelocity,
          worst(state.velocities, frame(velocities, step + 1)),
        );
      }
      // 🔴 The floor is SHAKE against SETTLE, not the tolerance. OpenMM solves
      // every closed triangle of three atoms each holding two constraints --
      // rigid water, which is 88% of the clusters here -- with SETTLE, an
      // analytic rigid-body solution, and only the rest with CCMA. SETTLE and
      // SHAKE both satisfy the constraints and land on *different points* of
      // the constraint manifold, about 1.8e-5 nm apart per step. Tightening our
      // tolerance from 1e-6 to 1e-12 does not move it at all, which is how that
      // was diagnosed. Implementing SETTLE is what tightens this; see
      // docs/DYNAMICS.md and HANDOFF.md.
      expect(worstPosition).toBeLessThan(5e-5);
      expect(worstVelocity).toBeLessThan(5e-2);
    });

    it("reproduces OpenMM's LFMiddle step from each frame", () => {
      const positions = load("lfmiddle", "positions");
      const velocities = load("lfmiddle", "velocities");
      const forces = load("lfmiddle", "forces");
      let worstPosition = 0;
      let worstVelocity = 0;
      for (let step = 0; step < steps; step++) {
        const state: IntegratorState = {
          positions: Float64Array.from(frame(positions, step)),
          velocities: Float64Array.from(frame(velocities, step)),
        };
        const result = langevinMiddleStep(f!.system, state, frame(forces, step), {
          timeStep: TIME_STEP,
          temperature: 300,
          friction: 0,
          constraintTolerance: TOLERANCE,
        });
        expect(result.positionConstraints.converged).toBe(true);
        expect(result.velocityConstraints?.converged).toBe(true);
        worstPosition = Math.max(
          worstPosition,
          worst(state.positions, frame(positions, step + 1)),
        );
        worstVelocity = Math.max(
          worstVelocity,
          worst(state.velocities, frame(velocities, step + 1)),
        );
      }
      // Same floor as the Verlet case, and the same cause.
      expect(worstPosition).toBeLessThan(5e-5);
      expect(worstVelocity).toBeLessThan(5e-2);
    });

    it("refuses to invent randomness when the thermostat is on", () => {
      const state: IntegratorState = {
        positions: Float64Array.from(f!.state.positions),
        velocities: new Float64Array(stride),
      };
      expect(() =>
        langevinMiddleStep(f!.system, state, new Float64Array(stride), {
          timeStep: TIME_STEP,
          temperature: 300,
          friction: 1,
          constraintTolerance: TOLERANCE,
        }),
      ).toThrow(/record its seed/);
    });

    it("leaves a frozen particle alone", () => {
      // Nothing in this workflow freezes particles, but an uploaded system may,
      // and a zero mass must mean zero motion rather than a division by zero.
      const frozen = {
        ...f!.system,
        masses: Float64Array.from(f!.system.masses, (mass, i) =>
          i === 0 ? 0 : mass,
        ),
      };
      const state: IntegratorState = {
        positions: Float64Array.from(f!.state.positions),
        velocities: new Float64Array(stride).fill(0.1),
      };
      const forces = new Float64Array(stride).fill(100);
      verletStep(frozen, state, forces, TIME_STEP, TOLERANCE);
      for (let axis = 0; axis < 3; axis++) {
        expect(state.positions[axis]).toBe(f!.state.positions[axis]);
      }
    });
  });
}
