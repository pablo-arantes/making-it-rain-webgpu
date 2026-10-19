/**
 * The minimiser against OpenMM's own, and the objective against itself.
 *
 * Energies at *matched iteration counts*, not just at the end. A minimum is not
 * unique: from a high-energy start, two minimisers with the same objective but
 * different line-search arithmetic drift into different basins, so comparing
 * converged energies says little. For a small number of iterations the path is
 * still shared -- L-BFGS's first direction is the gradient and the backtracking
 * is deterministic -- so those frames compare tightly, and what they pin is the
 * objective, which is what decides where the minimum is at all.
 *
 * 🔴 OpenMM does not reach its own tolerance on this fixture, and that is not a
 * defect in the capture. The documented default is 10 kJ/mol/nm on the
 * root-mean-square force, and the converged frame sits at 1080. The reason is
 * in `ReferenceMinimize.cpp`: the outer loop **ignores libLBFGS's return code**
 * and breaks as soon as the constraints are satisfied, so a stalled line search
 * ends the minimisation silently. With constraint springs at 1e6 kJ/mol/nm^2
 * against soft modes around 100, the objective is stiff enough that the search
 * does stall. The energy still falls by 16,221 kJ/mol, which is the part that
 * matters; the tolerance is aspirational.
 *
 * Anyone comparing against this should know the springs are *not* what keeps the
 * gradient large. At the end the constraint error is 1e-5 relative, so
 * `dr` is 1e-6 nm and the spring force is 1 kJ/mol/nm against a gradient of
 * 1000 -- the objective gradient and the raw potential gradient agree to four
 * figures. The minimisation genuinely stops short of a stationary point.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { evaluatePotential } from "../src/reference/potential.js";
import { minimise, maxConstraintError } from "../src/reference/minimise.js";
import { splitConstraints } from "../src/engine/constraint-clusters.js";
import type { Vec3 } from "../src/engine/system.js";
import { fixture } from "./fixtures.js";

interface MinimiseFacts {
  readonly particles: number;
  readonly spring_constant: number;
  readonly tolerance_kJ_mol_nm: number;
  readonly constraint_tolerance: number;
  readonly start: { readonly energy_kJ_mol: number; readonly max_constraint_error: number };
  readonly frames: Record<
    string,
    {
      readonly energy_kJ_mol: number;
      readonly objective_gradient_rms_per_particle: number;
      readonly max_constraint_error: number;
    }
  >;
}

const f = fixture("1L2Y");
const factsPath = "fixtures/1L2Y/minimise-facts.json";
const captured = existsSync(factsPath);

describe.skipIf(f === null || !captured)("minimise, against OpenMM on 1L2Y", () => {
  const facts = captured
    ? (JSON.parse(readFileSync(factsPath, "utf8")) as MinimiseFacts)
    : undefined;
  // The Reference platform uses the exact minimum mesh, not a smooth one, so a
  // comparison has to use the same mesh or the reciprocal energies differ.
  const grid = f?.facts.pme.grid_reference_platform as unknown as Vec3;
  const split = f === null ? undefined : splitConstraints(f.system);

  const potential = async (positions: Float64Array) => {
    const result = evaluatePotential(f!.system, positions, f!.state.boxVectors, {
      grid,
    });
    return { energy: result.energy, forces: result.forces };
  };

  it("agrees with OpenMM's energy at matched iteration counts", async () => {
    const reported: string[] = [];
    for (const key of ["1", "2", "5"]) {
      const expected = facts!.frames[key]!;
      const result = await minimise(
        f!.system,
        f!.state.positions,
        potential,
        {
          tolerance: facts!.tolerance_kJ_mol_nm,
          maxIterations: Number(key),
          constraintTolerance: facts!.constraint_tolerance,
          ...(split === undefined ? {} : { split }),
        },
      );
      const relative =
        Math.abs(result.energy - expected.energy_kJ_mol) /
        Math.abs(expected.energy_kJ_mol);
      reported.push(
        `  ${key.padStart(3)} iterations: ours ${result.energy.toFixed(4)}, ` +
          `OpenMM ${expected.energy_kJ_mol.toFixed(4)}, relative ${relative.toExponential(2)}` +
          `  |f| rms ${result.potentialForceRms.toFixed(2)} vs ` +
          `${expected.objective_gradient_rms_per_particle.toFixed(2)}`,
      );
      // Measured at 9.2e-5, 3.6e-5 and 4.8e-5 for one, two and five
      // iterations. The two searches are not bit-identical: our forces differ
      // from OpenMM's at 1e-7 relative, and near a backtracking decision that
      // is enough to flip one halving of the step, after which the paths differ
      // slightly for good. 2e-4 leaves room for that and would still catch a
      // wrong objective, which would be out by orders of magnitude -- a missing
      // constraint spring term alone is thousands of kJ/mol.
      expect(relative, reported.join("\n")).toBeLessThan(2e-4);
    }
    process.stderr.write(`\n${reported.join("\n")}\n`);
  }, 120_000);

  it("minimises a freshly built system into a usable one", async () => {
    // One run, several claims, because each run is 50-odd double-precision PME
    // evaluations and the claims are about the same result.
    const result = await minimise(f!.system, f!.state.positions, potential, {
      maxIterations: 20,
      constraintTolerance: 1e-6,
      ...(split === undefined ? {} : { split }),
    });
    process.stderr.write(
      `\n  20 iterations: energy ${result.energy.toFixed(2)} kJ/mol ` +
        `(from ${facts!.start.energy_kJ_mol.toFixed(2)}), ` +
        `potential |f| rms ${result.potentialForceRms.toFixed(2)}, ` +
        `constraint error ${result.maxConstraintError.toExponential(2)} ` +
        `-> ${maxConstraintError(f!.system, result.positions).toExponential(2)} ` +
        `after the final projection, ${result.rounds} rounds, k up to ` +
        `${result.restraintStrength.toExponential(0)}\n`,
    );

    // k = 100 / max(1e-4, constraintTolerance): the 1e-4 floor wins for the
    // notebook's 1e-6, so the springs start at 1e6 kJ/mol/nm^2 and not 1e8.
    expect(result.initialRestraintStrength).toBe(facts!.spring_constant);

    // A freshly built system violates its own constraints by 25%. After
    // minimisation they hold to the tolerance the caller asked for -- delivered
    // by the final exact projection, not by the springs, which only ever get
    // them to the 1e-4 working tolerance.
    expect(facts!.start.max_constraint_error).toBeGreaterThan(0.2);
    expect(result.maxConstraintError).toBeLessThan(1e-4);
    expect(maxConstraintError(f!.system, result.positions)).toBeLessThan(1e-6);

    expect(result.energy).toBeLessThan(facts!.start.energy_kJ_mol - 5000);
  }, 300_000);

  it("takes more rounds than OpenMM when given very few iterations", async () => {
    // Not a defect, and worth pinning so it is not read as one. With
    // `maxIterations: 1` the whole minimisation is a single step, and whether
    // that step leaves the constraints inside the 1e-4 working tolerance comes
    // down to one backtracking decision -- which our forces, 1e-7 from
    // OpenMM's, can land on the other side of. OpenMM finishes in one round at
    // 4.2e-5; we stiffen the springs a few times and finish at 5.9e-5. Both
    // satisfy the tolerance, and the energies still agree to 1e-4.
    const result = await minimise(f!.system, f!.state.positions, potential, {
      maxIterations: 1,
      constraintTolerance: 1e-6,
      ...(split === undefined ? {} : { split }),
    });
    expect(result.rounds).toBeGreaterThanOrEqual(1);
    expect(result.maxConstraintError).toBeLessThan(1e-4);
    expect(result.gaveUp).toBe(false);
  }, 300_000);
});
