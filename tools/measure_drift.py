"""OpenMM's own energy drift on a captured configuration.

Our Phase 1 gate is an energy-drift number, and a threshold for it has to be
calibrated rather than chosen (AGENTS.md).  The question is not "is this drift
small" in the abstract: a 1 nm cutoff with no switching function truncates the
Lennard-Jones force discontinuously, which drifts on its own, in OpenMM as much
as here.  So the target is OpenMM running the same system from the same
coordinates with a comparable precision.

Platforms, and why this one.  `Reference` is double throughout and would set a
target no single-precision engine can meet.  `CUDA`/`OpenCL` mixed precision is
the real target, and the right place to measure it is a GPU -- this Mac has no
working OpenCL (see the note in CLAUDE.md), so that run belongs on Colab.
`CPU` computes forces in single precision and accumulates in double, which is
the closest available comparison and is what this measures by default.

The drift is reported as a least-squares slope over the whole run, not as the
difference between the first and last report: over a short run the fluctuation
is comparable to the trend, and an endpoint difference cannot tell them apart.

Run with the oracle environment (tools/README.md):

    python tools/measure_drift.py --fixture fixtures/3HTB --steps 500
"""

import argparse
import json
import pathlib

import numpy as np
import openmm
from openmm import unit


def kinetic_with_shift(context, system, time_shift):
    """The kinetic energy a leapfrog integrator's conserved quantity needs.

    OpenMM's `VerletIntegrator` reports the energy of velocities shifted half a
    step forward and re-projected onto the constraint surface; the unshifted
    energy oscillates at O(dt) instead of being conserved, which would swamp any
    drift.  `getState(getEnergy=True)` on a Verlet context already applies the
    shift, so this is only needed when driving the integrator by hand.
    """
    state = context.getState(getEnergy=True)
    return state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=pathlib.Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--reports", type=int, default=20)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--platform", default="CPU")
    args = parser.parse_args()

    system = openmm.XmlSerializer.deserialize(
        (args.fixture / "system.xml").read_text()
    )
    # The captured frame, not state.xml: that state is un-minimised, and a drift
    # measured from a 5.6e5 kJ/mol/nm force means nothing.
    # Saved as (frames, 3N); frame 0 is the equilibrated start.
    positions = np.load(args.fixture / "trajectory-verlet-positions.npy").ravel()
    velocities = np.load(args.fixture / "trajectory-verlet-velocities.npy").ravel()
    particles = system.getNumParticles()
    start_x = positions[: particles * 3].reshape(particles, 3)
    start_v = velocities[: particles * 3].reshape(particles, 3)

    integrator = openmm.VerletIntegrator(args.timestep * unit.picosecond)
    integrator.setConstraintTolerance(1e-6)
    platform = openmm.Platform.getPlatformByName(args.platform)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(start_x * unit.nanometer)
    context.setVelocities(start_v * (unit.nanometer / unit.picosecond))

    per_report = max(1, args.steps // args.reports)
    times, totals = [], []
    for report in range(args.reports + 1):
        state = context.getState(getEnergy=True)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        times.append(report * per_report * args.timestep / 1000.0)  # ns
        totals.append(potential + kinetic)
        if report < args.reports:
            integrator.step(per_report)

    # Least squares slope, kJ/mol per ns, then per atom.
    slope = float(np.polyfit(np.array(times), np.array(totals), 1)[0])
    report = {
        "fixture": args.fixture.name,
        "platform": args.platform,
        "openmm": openmm.version.version,
        "particles": particles,
        "steps": args.steps,
        "timestep_ps": args.timestep,
        "picoseconds": args.steps * args.timestep,
        "total_energy_kJ_mol": totals,
        "times_ns": times,
        "endpoint_change_kJ_mol": totals[-1] - totals[0],
        "slope_kJ_mol_per_ns": slope,
        "drift_per_atom_per_ns": abs(slope) / particles,
        "note": (
            "VerletIntegrator reports the half-step-shifted kinetic energy, "
            "which is the conserved quantity; the unshifted one oscillates at "
            "O(dt). The slope is a least-squares fit, because over a short run "
            "the fluctuation is comparable to the trend."
        ),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
