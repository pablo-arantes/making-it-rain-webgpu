"""OpenMM's own minimisation, recorded step by step in iteration count.

Why several iteration counts rather than just the converged answer: a minimum is
not unique.  From a high-energy start two minimisers with the same objective but
different line-search arithmetic will drift into different basins, so comparing
only the final energies says little.  For a *small* number of iterations the
path is still shared -- L-BFGS's first direction is the gradient and the
backtracking is deterministic -- so those frames compare tightly and pin the
objective, which is the part that decides where the minimum is.

The converged frame is captured too, but as a loose check: the energy should be
far below the start and the forces below the tolerance.

Platform `Reference`, because this is the oracle for the double-precision
TypeScript reference.  That also means the PME mesh is the exact minimum rather
than a smooth size (`facts.json` records both), and a comparison has to pass the
same mesh or the reciprocal energies will not match.

    python tools/capture_minimisation.py --fixture fixtures/1L2Y
"""

import argparse
import json
import pathlib

import numpy as np
import openmm
from openmm import unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=pathlib.Path)
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--tolerance", type=float, default=10.0)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[1, 2, 5, 20, 100],
        help="maxIterations values to capture, plus the converged run",
    )
    args = parser.parse_args()

    system = openmm.XmlSerializer.deserialize(
        (args.fixture / "system.xml").read_text()
    )
    state_xml = (args.fixture / "state.xml").read_text()
    platform = openmm.Platform.getPlatformByName(args.platform)
    particles = system.getNumParticles()

    def fresh():
        """A context at the fixture's own starting coordinates.

        The integrator's constraint tolerance is what sets the minimiser's
        spring strength (k = 100/max(1e-4, tol)), so it has to be set even
        though no dynamics is run.
        """
        integrator = openmm.VerletIntegrator(0.002 * unit.picosecond)
        integrator.setConstraintTolerance(args.constraint_tolerance)
        context = openmm.Context(system, integrator, platform)
        context.setState(openmm.XmlSerializer.deserialize(state_xml))
        return context

    masses = np.array(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(particles)
        ]
    )
    spring_constant = 100.0 / max(1e-4, args.constraint_tolerance)

    def measure(context):
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

        # The gradient the minimiser actually tests, which is **not** minus the
        # potential force.  At a constrained minimum the potential gradient
        # balances the constraint forces and so is not small at all -- which is
        # why the raw force RMS below barely falls while the energy drops by
        # thousands.  OpenMM's objective adds the constraint springs, and their
        # gradient is what cancels it.  Transcribed from `evaluate` in
        # ReferenceMinimize.cpp.
        gradient = -forces.copy()
        gradient[masses == 0.0] = 0.0
        restraint_energy = 0.0
        worst = 0.0
        for c in range(system.getNumConstraints()):
            p1, p2, distance = system.getConstraintParameters(c)
            distance = distance.value_in_unit(unit.nanometer)
            delta = positions[p2] - positions[p1]
            r = float(np.linalg.norm(delta))
            unit_vector = delta / r
            dr = r - distance
            kdr = spring_constant * dr
            restraint_energy += 0.5 * kdr * dr
            if masses[p1] != 0.0:
                gradient[p1] -= kdr * unit_vector
            if masses[p2] != 0.0:
                gradient[p2] += kdr * unit_vector
            worst = max(worst, abs(dr) / distance)

        return {
            "energy_kJ_mol": state.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            ),
            "restraint_energy_kJ_mol": restraint_energy,
            "max_force_component": float(np.abs(forces).max()),
            "force_rms_per_particle": float(
                np.sqrt((forces ** 2).sum() / particles)
            ),
            # The convergence criterion, written out: OpenMM passes
            # epsilon = tolerance/norm with norm = sqrt(mean |x_i|^2) clamped at
            # 1, and libLBFGS tests |g|/max(|x|,1), which together reduce to
            # sqrt(sum |g|^2 / N) <= tolerance.
            "objective_gradient_rms_per_particle": float(
                np.sqrt((gradient ** 2).sum() / particles)
            ),
            "max_constraint_error": worst,
        }, positions

    report = {
        "fixture": args.fixture.name,
        "platform": args.platform,
        "openmm": openmm.version.version,
        "particles": particles,
        "constraints": system.getNumConstraints(),
        "tolerance_kJ_mol_nm": args.tolerance,
        "constraint_tolerance": args.constraint_tolerance,
        "spring_constant": 100.0 / max(1e-4, args.constraint_tolerance),
        "frames": {},
    }

    context = fresh()
    start, start_positions = measure(context)
    report["start"] = start
    np.save(args.fixture / "minimise-start-positions.npy", start_positions.ravel())

    for max_iterations in [*args.iterations, 0]:
        context = fresh()
        openmm.LocalEnergyMinimizer.minimize(
            context, args.tolerance, max_iterations
        )
        frame, positions = measure(context)
        key = "converged" if max_iterations == 0 else str(max_iterations)
        report["frames"][key] = frame
        np.save(
            args.fixture / f"minimise-{key}-positions.npy", positions.ravel()
        )
        print(
            f"  {key:>9}  energy {frame['energy_kJ_mol']:16.4f}  "
            f"objective |g| rms {frame['objective_gradient_rms_per_particle']:11.4f}  "
            f"raw force rms {frame['force_rms_per_particle']:10.2f}  "
            f"constraint err {frame['max_constraint_error']:.3e}"
        )

    (args.fixture / "minimise-facts.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.fixture / 'minimise-facts.json'}")


if __name__ == "__main__":
    main()
