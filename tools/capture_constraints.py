#!/usr/bin/env python
"""Capture OpenMM's constraint solver and its integrator, as oracles.

Two things the engine has to reproduce exactly, and OpenMM exposes both
directly rather than only through a whole simulation:

  * `Context.applyConstraints(tolerance)` and `applyVelocityConstraints` --
    SHAKE and RATTLE on coordinates we choose, so the solver can be compared on
    its own rather than inside a trajectory.
  * two deterministic trajectories, because one cannot do both jobs.

    `LangevinIntegrator` with **friction = 0** has `vscale = exp(-dt*0) = 1`
    and `noisescale = sqrt(1 - 1) = 0`, so its thermostat is the identity and
    every step is reproducible without matching OpenMM's random stream. That
    makes it the oracle for comparing our LFMiddle implementation step by step.

    It is **not** an oracle for energy conservation. With friction zero,
    LFMiddle is `v += dt*F/m` then `x += dt*v` -- symplectic Euler, not
    velocity Verlet -- which conserves a *modified* Hamiltonian, so the naive
    potential-plus-kinetic energy oscillates at O(dt) amplitude instead of
    drifting. Measured on 1L2Y at 2 fs: it bounces over +/-0.02 kJ/mol an atom
    with no trend and does not shrink cleanly with the timestep, while velocity
    Verlet on the same configuration stays inside 3e-3 and falls with dt. That
    is by design; LFMiddle exists for thermostatted sampling, not for NVE.

    So `VerletIntegrator` is captured as well, and it is what the NVE drift
    gate uses.

    python tools/capture_constraints.py fixtures/1L2Y --out fixtures/1L2Y
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import openmm
from openmm import XmlSerializer, unit

KJ = unit.kilojoule_per_mole
NM = unit.nanometer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("fixture", help="directory holding system.xml and state.xml")
    p.add_argument("--out", default=None)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--perturbation", type=float, default=0.01,
                   help="nm, the scale of the random displacement to constrain away")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--timestep", type=float, default=0.002, help="ps")
    p.add_argument("--equilibrate-ps", type=float, default=10.0,
                   help="minimise and run this much NVT before capturing; 0 uses "
                        "the state file as given")
    p.add_argument("--minimise-steps", type=int, default=10000)
    args = p.parse_args()
    out = args.out or args.fixture

    system = XmlSerializer.deserialize(open(f"{args.fixture}/system.xml").read())
    state = XmlSerializer.deserialize(open(f"{args.fixture}/state.xml").read())
    positions = np.array(state.getPositions().value_in_unit(NM))
    reference = openmm.Platform.getPlatformByName("Reference")

    report: dict = {
        "tolerance": args.tolerance,
        "particles": system.getNumParticles(),
        "constraints": system.getNumConstraints(),
        "timestep_ps": args.timestep,
        "openmm": openmm.version.version,
    }

    # ---- 1. the constraint solver on its own -------------------------------
    # A seeded displacement, large enough that the solver has real work to do.
    generator = np.random.default_rng(20261019)
    displacement = generator.normal(0.0, args.perturbation, positions.shape)
    perturbed = positions + displacement

    integrator = openmm.VerletIntegrator(args.timestep)
    integrator.setConstraintTolerance(args.tolerance)
    context = openmm.Context(system, integrator, reference)
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    context.setPositions(perturbed * NM)
    context.applyConstraints(args.tolerance)
    constrained = np.array(
        context.getState(getPositions=True).getPositions().value_in_unit(NM))

    # Velocities: a seeded set, then RATTLE at the *constrained* positions.
    velocities = generator.normal(0.0, 0.5, positions.shape)
    context.setVelocities(velocities * NM / unit.picosecond)
    context.applyVelocityConstraints(args.tolerance)
    constrained_velocities = np.array(
        context.getState(getVelocities=True).getVelocities()
        .value_in_unit(NM / unit.picosecond))
    del context, integrator

    report["constraint_case"] = {
        "perturbation_nm": args.perturbation,
        "max_displacement_nm": float(np.abs(displacement).max()),
        "max_correction_nm": float(np.abs(constrained - perturbed).max()),
        "max_velocity_correction": float(
            np.abs(constrained_velocities - velocities).max()),
    }
    np.save(os.path.join(out, "constraint-input-positions.npy"), perturbed)
    np.save(os.path.join(out, "constraint-output-positions.npy"), constrained)
    np.save(os.path.join(out, "constraint-input-velocities.npy"), velocities)
    np.save(os.path.join(out, "constraint-output-velocities.npy"),
            constrained_velocities)

    # ---- 2. a deterministic trajectory ------------------------------------
    # Minimise and equilibrate first. The fixtures hold the structure as built,
    # whose largest force component is 5.6e5 kJ/mol/nm for the solvated
    # protein-ligand case, and 2 fs dynamics from there goes to NaN within a few
    # steps -- so an oracle captured from it is no oracle at all. The same
    # lesson is recorded in tools/calibrate_thresholds.py.
    equilibrated = positions
    if args.equilibrate_ps > 0:
        warmup = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, args.timestep * unit.picosecond)
        warmup.setConstraintTolerance(args.tolerance)
        warmup.setRandomNumberSeed(20261019)
        # The fastest platform available; the configuration only has to be
        # plausible, not reproducible, because what is captured is the
        # trajectory *from* it.
        for platform_name, properties in (("CUDA", {"Precision": "mixed"}),
                                          ("OpenCL", {"Precision": "single"}),
                                          ("CPU", {})):
            try:
                warm_context = openmm.Context(
                    system, warmup, openmm.Platform.getPlatformByName(platform_name),
                    properties)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("no platform could build a context")
        warm_context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        warm_context.setPositions(positions * NM)
        openmm.LocalEnergyMinimizer.minimize(
            warm_context, 10 * KJ / NM, args.minimise_steps)
        warm_context.setVelocitiesToTemperature(300 * unit.kelvin, 20261019)
        warmup.step(int(args.equilibrate_ps / args.timestep))
        warm_state = warm_context.getState(getPositions=True, getEnergy=True)
        equilibrated = np.array(warm_state.getPositions().value_in_unit(NM))
        report["equilibration"] = {
            "platform": warm_context.getPlatform().getName(),
            "minimise_steps": args.minimise_steps,
            "picoseconds": args.equilibrate_ps,
            "potential_energy_kJ_mol":
                warm_state.getPotentialEnergy().value_in_unit(KJ),
        }
        del warm_context, warmup

    # Thermal velocities, so the trajectory is a plausible one rather than a
    # relaxation from rest, and constrained so the first step is not dominated
    # by fixing the initial state.
    seed_integrator = openmm.VerletIntegrator(args.timestep * unit.picosecond)
    seed_integrator.setConstraintTolerance(args.tolerance)
    seed = openmm.Context(system, seed_integrator, reference)
    seed.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    seed.setPositions(equilibrated * NM)
    seed.setVelocitiesToTemperature(300 * unit.kelvin, 20261019)
    seed.applyConstraints(args.tolerance)
    seed.applyVelocityConstraints(args.tolerance)
    seeded = seed.getState(getPositions=True, getVelocities=True)
    start_positions = np.array(seeded.getPositions().value_in_unit(NM))
    start_velocities = np.array(
        seeded.getVelocities().value_in_unit(NM / unit.picosecond))
    del seed, seed_integrator
    np.save(os.path.join(out, "trajectory-start-positions.npy"), start_positions)
    np.save(os.path.join(out, "trajectory-start-velocities.npy"), start_velocities)

    def capture(label: str, integrator) -> dict:
        integrator.setConstraintTolerance(args.tolerance)
        context = openmm.Context(system, integrator, reference)
        context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
        context.setPositions(start_positions * NM)
        context.setVelocities(start_velocities * NM / unit.picosecond)
        frames_x = []
        frames_v = []
        energies = []
        for step in range(args.steps + 1):
            s = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
            frames_x.append(np.array(s.getPositions().value_in_unit(NM)))
            frames_v.append(
                np.array(s.getVelocities().value_in_unit(NM / unit.picosecond)))
            energies.append({
                "potential": s.getPotentialEnergy().value_in_unit(KJ),
                "kinetic": s.getKineticEnergy().value_in_unit(KJ),
            })
            if step < args.steps:
                integrator.step(1)
        del context, integrator
        np.save(os.path.join(out, f"trajectory-{label}-positions.npy"),
                np.array(frames_x))
        np.save(os.path.join(out, f"trajectory-{label}-velocities.npy"),
                np.array(frames_v))
        totals = [e["potential"] + e["kinetic"] for e in energies]
        if not np.isfinite(totals).all():
            raise SystemExit(
                f"the captured {label} trajectory is not finite. The starting "
                "structure is too strained for this timestep; raise "
                "--equilibrate-ps or --minimise-steps.")
        return {
            "steps": args.steps,
            "energies_kJ_mol": energies,
            "total_energy_change_kJ_mol": totals[-1] - totals[0],
            "total_energy_change_per_atom":
                (totals[-1] - totals[0]) / system.getNumParticles(),
        }

    report["trajectory_lfmiddle"] = {
        **capture("lfmiddle", openmm.LangevinIntegrator(
            300 * unit.kelvin, 0.0 / unit.picosecond,
            args.timestep * unit.picosecond)),
        "integrator": "LangevinIntegrator, friction 0 (= LFMiddle)",
        "purpose": "step-by-step determinism; NOT energy conservation, because "
                   "LFMiddle at zero friction is symplectic Euler and its naive "
                   "total energy oscillates at O(dt)",
    }
    report["trajectory_verlet"] = {
        **capture("verlet", openmm.VerletIntegrator(args.timestep * unit.picosecond)),
        "integrator": "VerletIntegrator (velocity Verlet)",
        "purpose": "the NVE energy-drift gate, and step-by-step determinism",
    }
    with open(os.path.join(out, "dynamics-facts.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print(f"\nwrote constraint and trajectory arrays to {out}/")


if __name__ == "__main__":
    main()
