#!/usr/bin/env python
"""Host-side anchors for the WebGPU engine: Reference energies and ns/day.

Reports, for a fixture built by tools/build_reference_system.py:
  * per-force-group potential energies from the double-precision Reference
    platform (the acceptance oracle for the WGSL kernels), and
  * ns/day for the notebook's production settings on every OpenMM platform
    this host offers, so a WebGPU number can be compared with something real
    on the same machine.

    python tools/bench_reference_host.py fixtures/3HTB/system.xml \
        fixtures/3HTB/state.xml --out docs/measurements/host-timings.json
"""
import argparse
import json
import platform
import time

import openmm
from openmm import XmlSerializer, unit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("system_xml")
    p.add_argument("state_xml")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--minimise-steps", type=int, default=10000)
    p.add_argument("--out")
    args = p.parse_args()

    system = XmlSerializer.deserialize(open(args.system_xml).read())
    state = XmlSerializer.deserialize(open(args.state_xml).read())
    out = {
        "host": {"platform": platform.platform(), "machine": platform.machine(),
                 "processor": platform.processor()},
        "openmm": openmm.version.version,
        "atoms": system.getNumParticles(),
        "available_platforms": [openmm.Platform.getPlatform(i).getName()
                                for i in range(openmm.Platform.getNumPlatforms())],
    }
    positions = state.getPositions()

    # minimise once, as the notebook's equilibration does, so dynamics is stable
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.002),
                         openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(positions)
    t0 = time.time()
    openmm.LocalEnergyMinimizer.minimize(
        ctx, 10 * unit.kilojoule_per_mole / unit.nanometer, args.minimise_steps)
    minimised = ctx.getState(getPositions=True, getEnergy=True)
    out["minimisation"] = {
        "platform": "CPU", "max_iterations": args.minimise_steps,
        "seconds": round(time.time() - t0, 1),
        "energy_kJ_mol": minimised.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
    }
    positions = minimised.getPositions()
    del ctx

    out["ns_per_day"] = {}
    for name, props in (("CUDA", {"Precision": "mixed"}), ("OpenCL", {"Precision": "mixed"}),
                        ("HIP", {"Precision": "mixed"}), ("CPU", {})):
        try:
            plat = openmm.Platform.getPlatformByName(name)
        except Exception:
            continue
        s = XmlSerializer.deserialize(open(args.system_xml).read())
        s.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 298 * unit.kelvin))
        integ = openmm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                          0.002 * unit.picosecond)
        integ.setConstraintTolerance(1e-6)
        try:
            c = openmm.Context(s, integ, plat, props)
        except Exception as exc:
            out["ns_per_day"][name] = f"unavailable: {exc}"
            continue
        c.setPositions(positions)
        c.setVelocitiesToTemperature(298 * unit.kelvin)
        integ.step(200)
        steps = args.steps if name != "CPU" else max(400, args.steps // 8)
        t0 = time.time()
        integ.step(steps)
        c.getState(getEnergy=True).getPotentialEnergy()
        dt = time.time() - t0
        out["ns_per_day"][name] = {
            "steps": steps, "timestep_fs": 2, "seconds": round(dt, 2),
            "ns_per_day": round(steps * 0.002e-3 / (dt / 86400.0), 1),
        }
        print(name, out["ns_per_day"][name])
        del c, integ, s
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
