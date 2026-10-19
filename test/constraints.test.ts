import { existsSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  applyPositionConstraints,
  applyVelocityConstraints,
  inverseMasses,
} from "../src/reference/constraints.js";
import { readNpy } from "./npy.js";
import { fixture, FIXTURE_NAMES } from "./fixtures.js";

/**
 * SHAKE and RATTLE against OpenMM's own solver.
 *
 * `tools/capture_constraints.py` perturbs the fixture by a seeded 0.01 nm
 * displacement -- enough to put the worst constraint 57% out -- and records what
 * `Context.applyConstraints` does with it. That compares the solver on its own,
 * rather than only inside a trajectory where a constraint bug and an integrator
 * bug look alike.
 *
 * 🔴 `Context.applyConstraints` passes the *same array* as both the reference
 * and the trial: the correction directions are the current bond vectors, not
 * the bond vectors from before a step. That is not what SHAKE does inside an
 * integrator, where the reference is the previous step's positions. Getting
 * this wrong put our answer 0.04 nm from OpenMM's while every constraint was
 * still satisfied, so the solver looked right and the comparison looked broken.
 */
for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  const captured = existsSync(`fixtures/${name}/constraint-input-positions.npy`);
  if (f !== null && !captured) {
    process.stderr.write(
      `\n  fixtures/${name} has no constraint capture. Generate it with:\n` +
        `    python tools/capture_constraints.py fixtures/${name}\n\n`,
    );
  }

  describe.skipIf(f === null || !captured)(`constraints for ${name}`, () => {
    const TOLERANCE = 1e-6;
    const read = (file: string): Float64Array =>
      readNpy(`fixtures/${name}/${file}`).data;

    it("satisfies every constraint to the tolerance", () => {
      const trial = Float64Array.from(read("constraint-input-positions.npy"));
      const result = applyPositionConstraints(
        f!.system,
        f!.state.positions,
        trial,
        TOLERANCE,
      );
      expect(result.converged).toBe(true);
      // Relative, as OpenMM's tolerance is.
      expect(result.worstRelativeError).toBeLessThan(TOLERANCE);
    });

    it("converges in a handful of iterations at a realistic step size", () => {
      // The stress case above displaces atoms by 0.01 nm, which is a hundred
      // times what a 2 fs step moves them, and plain SHAKE needs 89 iterations
      // on 1L2Y and hits the 150 cap on 3HTB there. What matters for a running
      // simulation is this case.
      let seed = 7;
      const random = (): number =>
        (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
      const trial = Float64Array.from(f!.state.positions, (value) =>
        value + (random() * 2 - 1) * 1e-4,
      );
      const result = applyPositionConstraints(
        f!.system,
        f!.state.positions,
        trial,
        TOLERANCE,
      );
      expect(result.converged).toBe(true);
      // Recorded rather than asserted tightly: the point is the order of
      // magnitude, and that it is nowhere near the 150-iteration cap.
      expect(result.iterations).toBeLessThan(40);
    });

    it("lands where OpenMM's solver lands", () => {
      const perturbed = read("constraint-input-positions.npy");
      const trial = Float64Array.from(perturbed);
      const want = read("constraint-output-positions.npy");
      // Reference *and* trial are the perturbed positions, which is what
      // Context.applyConstraints does.
      applyPositionConstraints(f!.system, perturbed, trial, TOLERANCE);

      let worst = 0;
      for (let i = 0; i < want.length; i++) {
        worst = Math.max(worst, Math.abs((trial[i] as number) - (want[i] as number)));
      }
      // Both stop as soon as every constraint is inside a *relative* 1e-6, so
      // they stop at different points inside that band. On a 0.1 nm bond that
      // band is 1e-7 nm wide, which is the floor here -- it is set by OpenMM's
      // stopping rule, not by our solver. Measured: 1.8e-7 nm on 1L2Y and
      // 6.2e-7 on 3HTB.
      expect(worst).toBeLessThan(2e-6);
    });

    it("moves atoms only along the reference bonds, conserving momentum", () => {
      const before = read("constraint-input-positions.npy");
      const trial = Float64Array.from(before);
      applyPositionConstraints(f!.system, before, trial, TOLERANCE);
      // Every correction is `r_ij * delta / m`, equal and opposite on the two
      // atoms, so the mass-weighted displacement sums to zero.
      const momentum = [0, 0, 0];
      for (let atom = 0; atom < f!.system.particleCount; atom++) {
        const mass = f!.system.masses[atom] as number;
        for (let axis = 0; axis < 3; axis++) {
          const i = atom * 3 + axis;
          momentum[axis] =
            (momentum[axis] as number) +
            mass * ((trial[i] as number) - (before[i] as number));
        }
      }
      const scale = f!.system.particleCount;
      for (const component of momentum) {
        expect(Math.abs(component) / scale).toBeLessThan(1e-12);
      }
    });

    it("removes the velocity along every constrained bond", () => {
      const velocities = Float64Array.from(read("constraint-input-velocities.npy"));
      const positions = read("constraint-output-positions.npy");
      const result = applyVelocityConstraints(
        f!.system,
        positions,
        velocities,
        TOLERANCE,
      );
      expect(result.converged).toBe(true);

      // The physical statement: d/dt of each constrained distance is zero.
      //
      // The bound is the convergence criterion's own, not an arbitrary number.
      // RATTLE stops when |delta| <= tolerance with
      // delta = -2 * reducedMass * (r . v) / |r|^2, so |r . v| is allowed up to
      // tolerance * |r|^2 / (2 * reducedMass). Over these systems the loosest
      // constraint is water's H-H, which allows 4.5e-8 nm^2/ps.
      let allowed = 0;
      const inverse = inverseMasses(f!.system);
      for (let c = 0; c < f!.system.constraintCount; c++) {
        const reduced =
          0.5 /
          ((inverse[f!.system.constraintP1[c] as number] as number) +
            (inverse[f!.system.constraintP2[c] as number] as number));
        const d = f!.system.constraintDistance[c] as number;
        allowed = Math.max(allowed, (TOLERANCE * d * d) / (2 * reduced));
      }
      let worst = 0;
      for (let c = 0; c < f!.system.constraintCount; c++) {
        const p1 = (f!.system.constraintP1[c] as number) * 3;
        const p2 = (f!.system.constraintP2[c] as number) * 3;
        let projection = 0;
        for (let axis = 0; axis < 3; axis++) {
          projection +=
            ((positions[p1 + axis] as number) - (positions[p2 + axis] as number)) *
            ((velocities[p1 + axis] as number) - (velocities[p2 + axis] as number));
        }
        worst = Math.max(worst, Math.abs(projection));
      }
      expect(worst).toBeLessThanOrEqual(allowed);
    });

    it("lands where OpenMM's velocity solver lands", () => {
      const velocities = Float64Array.from(read("constraint-input-velocities.npy"));
      const positions = read("constraint-output-positions.npy");
      const want = read("constraint-output-velocities.npy");
      applyVelocityConstraints(f!.system, positions, velocities, TOLERANCE);
      let worst = 0;
      for (let i = 0; i < want.length; i++) {
        worst = Math.max(
          worst,
          Math.abs((velocities[i] as number) - (want[i] as number)),
        );
      }
      expect(worst).toBeLessThan(1e-6);
    });

    it("has real work to do on the state as built", () => {
      // 🔴 A freshly built system does NOT satisfy its own constraints. Measured
      // on both fixtures: the worst constrained distance is 25% away from the
      // force field's value, because `Modeller.addHydrogens` places hydrogens
      // geometrically and then relaxes them for only 50 iterations at a
      // 1.0 kJ/mol/nm tolerance (see docs/NOTEBOOK_PARITY.md finding 2). SHAKE
      // moves atoms up to 0.02 nm to fix it.
      //
      // So dynamics has to start with a constraint projection. The notebook
      // gets that implicitly from minimizeEnergy; the engine must do it
      // explicitly.
      let worstBefore = 0;
      for (let c = 0; c < f!.system.constraintCount; c++) {
        const p1 = (f!.system.constraintP1[c] as number) * 3;
        const p2 = (f!.system.constraintP2[c] as number) * 3;
        let squared = 0;
        for (let axis = 0; axis < 3; axis++) {
          squared +=
            ((f!.state.positions[p1 + axis] as number) -
              (f!.state.positions[p2 + axis] as number)) **
            2;
        }
        worstBefore = Math.max(
          worstBefore,
          Math.abs(
            Math.sqrt(squared) / (f!.system.constraintDistance[c] as number) - 1,
          ),
        );
      }
      expect(worstBefore).toBeGreaterThan(0.1);

      const trial = Float64Array.from(f!.state.positions);
      const result = applyPositionConstraints(
        f!.system,
        f!.state.positions,
        trial,
        TOLERANCE,
      );
      expect(result.converged).toBe(true);
      let moved = 0;
      for (let i = 0; i < trial.length; i++) {
        moved = Math.max(
          moved,
          Math.abs((trial[i] as number) - (f!.state.positions[i] as number)),
        );
      }
      expect(moved).toBeGreaterThan(1e-3);
      expect(moved).toBeLessThan(0.1);
    });

    it("gives every constrained particle a finite inverse mass", () => {
      const inverse = inverseMasses(f!.system);
      for (let c = 0; c < f!.system.constraintCount; c++) {
        const a = inverse[f!.system.constraintP1[c] as number] as number;
        const b = inverse[f!.system.constraintP2[c] as number] as number;
        // A constraint between two frozen particles could never be satisfied.
        expect(a + b).toBeGreaterThan(0);
      }
    });
  });
}
