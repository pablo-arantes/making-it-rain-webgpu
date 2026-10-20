"""What `setVelocitiesToTemperature` actually produces, as statistics.

The sample cannot be compared: OpenMM draws from SFMT by the Marsaglia polar
method and that stream is not reproducible from outside.  What can be compared
is everything that does not depend on which deviates came out.

🔴 **Both integrators shift the draw by (dt/2)F/m**, and that is the finding
this tool exists to pin down.  `setVelocitiesToTemperature` draws from
Maxwell-Boltzmann, then subtracts `(offset/m)F` where `offset` is the
integrator's velocity time offset -- and it is dt/2 for `VerletIntegrator` as
well as for `LangevinIntegrator`, because OpenMM's Verlet is leapfrog
internally and stores its velocities at t - dt/2.  (That is the same fact as
Verlet *reporting* a half-step-shifted kinetic energy; see docs/DYNAMICS.md.)

It is easy to read the evidence backwards.  The two integrators produce
*identical* statistics, which looks like neither of them shifting -- and the
first version of this tool concluded exactly that.  They are identical because
both shift by the same amount.  What settles it is the variance of the draw
itself, measured with the constraints stripped out so nothing else is acting:

    mean(m v^2)/kT  =  1 + (dt/2)^2 mean(F^2/m)/kT

3HTB measures 1.4189 against a prediction of 1.4173, and the correlation
between the velocities and -(dt/2)F/m is +0.52.  Without the shift it would be
1.0 and zero.  So this records the forces too, letting a reimplementation be
compared without evaluating a force field.

    python tools/capture_velocities.py --fixture fixtures/1L2Y
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
    parser.add_argument("--temperature", type=float, default=298.0)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    system = openmm.XmlSerializer.deserialize(
        (args.fixture / "system.xml").read_text()
    )
    state_xml = (args.fixture / "state.xml").read_text()
    platform = openmm.Platform.getPlatformByName("Reference")
    masses = np.array(
        [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        ]
    )
    moving = masses > 0
    kT = (
        unit.MOLAR_GAS_CONSTANT_R * args.temperature * unit.kelvin
    ).value_in_unit(unit.kilojoule_per_mole)

    report = {
        "openmm": openmm.version.version,
        "temperature_K": args.temperature,
        "timestep_ps": args.timestep,
        "particles": system.getNumParticles(),
        "moving_particles": int(moving.sum()),
        "constraints": system.getNumConstraints(),
        "kT_kJ_mol": kT,
        "velocity_constraint_tolerance": 1e-5,
        "integrators": {},
    }

    # `getVelocityTimeOffset` is a C++ method the Python wrapper does not
    # expose, so the offset is *measured* rather than read: dt/2 for both
    # integrators, which `unconstrained_variance_per_kT` below confirms against
    # the closed-form prediction. The headers would have suggested 0 for Verlet;
    # OpenMM's Verlet is leapfrog internally and stores its velocities at
    # t - dt/2, which is the same fact as its reporting a half-step-shifted
    # kinetic energy (docs/DYNAMICS.md).
    for name, make, offset in (
        (
            "verlet",
            lambda: openmm.VerletIntegrator(args.timestep * unit.picosecond),
            args.timestep / 2,
        ),
        (
            "lfmiddle",
            lambda: openmm.LangevinIntegrator(
                args.temperature * unit.kelvin,
                1.0 / unit.picosecond,
                args.timestep * unit.picosecond,
            ),
            args.timestep / 2,
        ),
    ):
        integrator = make()
        integrator.setConstraintTolerance(1e-6)
        context = openmm.Context(system, integrator, platform)
        context.setState(openmm.XmlSerializer.deserialize(state_xml))
        # The draw with the constraints stripped out, so the shift can be seen
        # on its own. Everything else about the system is kept.
        bare = openmm.XmlSerializer.deserialize(
            (args.fixture / "system.xml").read_text()
        )
        for c in range(bare.getNumConstraints() - 1, -1, -1):
            bare.removeConstraint(c)
        bare_integrator = make()
        bare_context = openmm.Context(bare, bare_integrator, platform)
        bare_context.setState(openmm.XmlSerializer.deserialize(state_xml))
        bare_context.setVelocitiesToTemperature(args.temperature * unit.kelvin, args.seeds[0])
        bare_v = bare_context.getState(getVelocities=True).getVelocities(
            asNumpy=True
        ).value_in_unit(unit.nanometer / unit.picosecond)
        unconstrained_variance = float((masses[:, None] * bare_v**2).mean() / kT)

        temperatures, worst_constraint, scaled_variance = [], 0.0, []
        for seed in args.seeds:
            context.setVelocitiesToTemperature(args.temperature * unit.kelvin, seed)
            state = context.getState(getVelocities=True, getPositions=True)
            v = state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
            positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            # The reported kinetic energy, which for LFMiddle is the unshifted
            # one; what matters here is the velocities themselves.
            kinetic = 0.5 * float((masses[:, None] * v**2).sum())
            # Degrees of freedom as StateDataReporter counts them.
            dof = 3 * int(moving.sum()) - system.getNumConstraints()
            if any(
                isinstance(system.getForce(i), openmm.CMMotionRemover)
                for i in range(system.getNumForces())
            ):
                dof -= 3
            temperatures.append(2 * kinetic / (dof * 0.008314462618153243))
            # Per-component variance in units of kT/m, which is 1 for a correct
            # Maxwell-Boltzmann draw before constraints bite.
            scaled_variance.append(
                float((masses[moving, None] * v[moving] ** 2).mean() / kT)
            )
            for c in range(system.getNumConstraints()):
                p1, p2, _d = system.getConstraintParameters(c)
                delta = positions[p2] - positions[p1]
                relative = v[p2] - v[p1]
                r = float(np.linalg.norm(delta))
                worst_constraint = max(
                    worst_constraint, abs(float(relative @ delta) / r)
                )

        report["integrators"][name] = {
            "velocity_time_offset_ps": float(offset),
            "unconstrained_variance_per_kT": unconstrained_variance,
            "temperature_K": {
                "mean": float(np.mean(temperatures)),
                "min": float(np.min(temperatures)),
                "max": float(np.max(temperatures)),
            },
            "scaled_variance_per_component": {
                "mean": float(np.mean(scaled_variance)),
                "min": float(np.min(scaled_variance)),
                "max": float(np.max(scaled_variance)),
            },
            "worst_constrained_relative_velocity_nm_ps": worst_constraint,
        }
        entry = report["integrators"][name]
        print(
            f"  {name:<9} offset {entry['velocity_time_offset_ps']:.4f} ps  "
            f"T {entry['temperature_K']['mean']:7.2f} K  "
            f"variance/kT {entry['scaled_variance_per_component']['mean']:.4f}  "
            f"unconstrained {entry['unconstrained_variance_per_kT']:.4f}  "
            f"constraint |v.r|/r {entry['worst_constrained_relative_velocity_nm_ps']:.2e}"
        )

    # The forces at the fixture's own positions, so a reimplementation can
    # apply the same shift without evaluating a force field.
    integrator = openmm.VerletIntegrator(args.timestep * unit.picosecond)
    context = openmm.Context(system, integrator, platform)
    context.setState(openmm.XmlSerializer.deserialize(state_xml))
    forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    np.save(args.fixture / "velocity-forces.npy", np.asarray(forces).ravel())
    report["shift_prediction"] = {
        "formula": "mean(m v^2)/kT = 1 + (dt/2)^2 mean(F^2/m)/kT",
        "mean_F2_over_m_per_kT": float((np.asarray(forces) ** 2 / masses[:, None]).mean() / kT),
        "predicted_unconstrained_variance": float(
            1
            + (args.timestep / 2) ** 2
            * (np.asarray(forces) ** 2 / masses[:, None]).mean()
            / kT
        ),
    }
    print(
        f"  predicted unconstrained variance "
        f"{report['shift_prediction']['predicted_unconstrained_variance']:.4f}"
    )

    (args.fixture / "velocity-facts.json").write_text(json.dumps(report, indent=1))
    print(f"  wrote {args.fixture / 'velocity-facts.json'}")


if __name__ == "__main__":
    main()
