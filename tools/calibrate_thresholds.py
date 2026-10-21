#!/usr/bin/env python
"""Calibrate the engine's acceptance thresholds against OpenMM's own platforms.

The WebGPU engine's target is not "agrees with double precision" -- nothing in
single precision does -- it is "no worse than OpenMM's mixed-precision GPU
platform, within a small factor".  That number has to be measured on a real GPU
rather than guessed, which is what this script is for.

For every platform and precision this host offers it reports, against the
double-precision Reference platform on identical coordinates:

  * per-force-group potential energy, absolute and relative;
  * per-atom force error -- RMS, maximum, and the relative RMS;
  * NVE energy drift with constraints at the notebook's 2 fs, in
    kJ/mol/ns/atom, which is the dynamics gate;
  * ns/day under the notebook's production settings.

The resulting JSON goes in docs/measurements/ and its numbers go in AGENTS.md
as the thresholds, naming the GPU that produced them.

Intended home: a Colab GPU runtime (T4), which is also where the notebook runs.

    !pip -q install condacolab && python -c "import condacolab; condacolab.install()"
    # after the kernel restarts:
    !mamba install -q -y -c conda-forge openmm
    !python tools/calibrate_thresholds.py fixtures/3HTB/system.xml \
        fixtures/3HTB/state.xml --out docs/measurements/calibration-T4.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import openmm
from openmm import XmlSerializer, unit

KJ = unit.kilojoule_per_mole
KJ_NM = unit.kilojoule_per_mole / unit.nanometer

# every (platform, precision) pair worth measuring; missing ones are skipped
CANDIDATES = [
    ("CUDA", "double"), ("CUDA", "mixed"), ("CUDA", "single"),
    ("HIP", "mixed"), ("OpenCL", "mixed"), ("OpenCL", "single"),
    ("CPU", None),
]


def group_the_forces(system: openmm.System) -> list[str]:
    """One force group per force, with PME reciprocal space split out."""
    names = []
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
        names.append(f.__class__.__name__)
    nb = next((f for f in system.getForces()
               if isinstance(f, openmm.NonbondedForce)), None)
    if nb is not None:
        nb.setReciprocalSpaceForceGroup(len(names))
        names.append("NonbondedForce[reciprocal]")
    return names


# Why each candidate platform could not be used, filled in by make_context.
# 🔴 Recorded, not just printed. A platform that fails to produce a Context was
# previously skipped with `continue` and a line of stdout, which left a JSON
# file that looks complete and is missing the only row that matters. One Colab
# run came back with CUDA in `available_platforms`, no CUDA row, and no record
# of the reason -- the cause existed only in a notebook cell that had since been
# cleared.
SKIPPED: dict[str, str] = {}


def make_context(system_xml: str, name: str, precision: str | None,
                 grouped: bool) -> tuple[openmm.Context, list[str]] | None:
    system = XmlSerializer.deserialize(open(system_xml).read())
    names = group_the_forces(system) if grouped else []
    label = f"{name}/{precision}" if precision else name
    try:
        plat = openmm.Platform.getPlatformByName(name)
    except Exception as exc:
        SKIPPED[label] = f"the platform is not registered: {exc}"
        return None
    props = {"Precision": precision} if precision else {}
    try:
        ctx = openmm.Context(system, openmm.VerletIntegrator(0.002), plat, props)
    except Exception as exc:
        # 🔴 The platform being *registered* does not mean it is usable. OpenMM
        # lists CUDA whenever its plugin loads, which needs only libcuda --
        # not a device, not a matching driver. So "CUDA is available" and "a
        # Context can be made on CUDA" are different claims, and this is where
        # the second one is tested.
        SKIPPED[label] = f"{type(exc).__name__}: {exc}"
        print(f"  {label}: unavailable ({exc})")
        return None
    return ctx, names


def single_point(ctx: openmm.Context, names: list[str], positions, box) -> dict:
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    energies = {}
    for g, n in enumerate(names):
        energies[f"{g}:{n}"] = ctx.getState(getEnergy=True, groups={g}) \
            .getPotentialEnergy().value_in_unit(KJ)
    state = ctx.getState(getEnergy=True, getForces=True)
    return {
        "energies": energies,
        "total": state.getPotentialEnergy().value_in_unit(KJ),
        "forces": state.getForces(asNumpy=True).value_in_unit(KJ_NM),
    }


def compare(ref: dict, got: dict) -> dict:
    fr, fg = ref["forces"], got["forces"]
    diff = fg - fr
    ref_norms = np.linalg.norm(fr, axis=1)
    out = {
        "energy_abs_kJ_mol": {k: got["energies"][k] - v for k, v in ref["energies"].items()},
        "energy_rel": {k: (got["energies"][k] - v) / abs(v) if abs(v) > 1e-9 else None
                       for k, v in ref["energies"].items()},
        "total_abs_kJ_mol": got["total"] - ref["total"],
        "total_rel": (got["total"] - ref["total"]) / abs(ref["total"]),
        "force_rms_error_kJ_mol_nm": float(np.sqrt((diff ** 2).sum(axis=1).mean())),
        "force_max_error_kJ_mol_nm": float(np.linalg.norm(diff, axis=1).max()),
        "force_rel_rms": float(np.sqrt((diff ** 2).sum() / (fr ** 2).sum())),
        "force_rms_reference_kJ_mol_nm": float(np.sqrt((fr ** 2).sum(axis=1).mean())),
        "worst_atom": int(np.linalg.norm(diff, axis=1).argmax()),
        "worst_atom_reference_force_kJ_mol_nm": float(
            ref_norms[int(np.linalg.norm(diff, axis=1).argmax())]),
    }
    return out


def equilibrate(system_xml: str, positions, box, picoseconds: float) -> tuple:
    """Minimise and run a short NVT, so comparisons are made at a configuration
    the dynamics would actually visit.

    Measured at the fixture's own starting coordinates the force error is
    dominated by a handful of clashing atoms, which makes a relative RMS say
    more about the input structure than about the platform.
    """
    system = XmlSerializer.deserialize(open(system_xml).read())
    integ = openmm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                      0.002 * unit.picosecond)
    integ.setConstraintTolerance(1e-6)
    integ.setRandomNumberSeed(1234)
    # the fastest platform available; the configuration only has to be plausible
    for name, props in (("CUDA", {"Precision": "mixed"}), ("HIP", {"Precision": "mixed"}),
                        ("OpenCL", {"Precision": "single"}), ("CPU", {})):
        try:
            ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName(name), props)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("no platform could build a context")
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    t0 = time.time()
    openmm.LocalEnergyMinimizer.minimize(ctx, 10 * KJ_NM, 10000)
    ctx.setVelocitiesToTemperature(298 * unit.kelvin, 1234)
    integ.step(int(picoseconds / 0.002))
    st = ctx.getState(getPositions=True, getVelocities=True, getEnergy=True)
    info = {"platform": ctx.getPlatform().getName(), "picoseconds": picoseconds,
            "seconds": round(time.time() - t0, 1),
            "potential_energy_kJ_mol": st.getPotentialEnergy().value_in_unit(KJ)}
    out = (st.getPositions(), st.getVelocities(), st.getPeriodicBoxVectors(), info)
    del ctx, integ
    return out


def nve_drift(system_xml: str, name: str, precision: str | None, positions, box,
              steps: int, report_every: int, velocities=None) -> dict | None:
    """Energy drift with constraints, no thermostat and no barostat."""
    made = make_context(system_xml, name, precision, grouped=False)
    if made is None:
        return None
    ctx, _ = made
    integ = ctx.getIntegrator()
    integ.setConstraintTolerance(1e-6)
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    if velocities is not None:
        ctx.setVelocities(velocities)
    else:
        ctx.setVelocitiesToTemperature(298 * unit.kelvin, 1234)
    ctx.applyConstraints(1e-6)
    n = ctx.getSystem().getNumParticles()
    series = []
    for i in range(0, steps, report_every):
        st = ctx.getState(getEnergy=True)
        series.append((i * 0.002,
                       (st.getPotentialEnergy() + st.getKineticEnergy()).value_in_unit(KJ)))
        integ.step(report_every)
    t = np.array([s[0] for s in series])          # ps
    e = np.array([s[1] for s in series])          # kJ/mol
    slope = float(np.polyfit(t, e, 1)[0]) if len(t) > 2 else float("nan")
    worst = float(np.abs(e - e[0]).max())
    del ctx
    return {
        "steps": steps, "timestep_fs": 2, "atoms": n,
        "drift_kJ_mol_per_ns_per_atom": slope * 1000.0 / n,
        "max_excursion_kJ_mol_per_atom": worst / n,
        "total_energy_series_kJ_mol": [round(v, 4) for v in e.tolist()],
    }


def speed(system_xml: str, name: str, precision: str | None, positions, box,
          steps: int) -> dict | None:
    system = XmlSerializer.deserialize(open(system_xml).read())
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 298 * unit.kelvin))
    integ = openmm.LangevinIntegrator(298 * unit.kelvin, 1.0 / unit.picosecond,
                                      0.002 * unit.picosecond)
    integ.setConstraintTolerance(1e-6)
    try:
        plat = openmm.Platform.getPlatformByName(name)
        ctx = openmm.Context(system, integ, plat,
                             {"Precision": precision} if precision else {})
    except Exception:
        return None
    ctx.setPeriodicBoxVectors(*box)
    ctx.setPositions(positions)
    ctx.setVelocitiesToTemperature(298 * unit.kelvin, 1234)
    integ.step(200)
    t0 = time.time()
    integ.step(steps)
    ctx.getState(getEnergy=True).getPotentialEnergy()
    dt = time.time() - t0
    del ctx, integ
    return {"steps": steps, "seconds": round(dt, 2),
            "ns_per_day": round(steps * 0.002e-3 / (dt / 86400.0), 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("system_xml")
    p.add_argument("state_xml")
    p.add_argument("--nve-steps", type=int, default=25000, help="50 ps at 2 fs")
    p.add_argument("--nve-report", type=int, default=500)
    p.add_argument("--speed-steps", type=int, default=5000)
    p.add_argument("--equilibrate-ps", type=float, default=10.0,
                   help="minimise and run this much NVT before comparing; 0 uses "
                        "the state file as given")
    p.add_argument("--skip-nve", action="store_true")
    p.add_argument("--skip-speed", action="store_true")
    p.add_argument("--out")
    p.add_argument(
        "--save-equilibrated",
        help="write the equilibrated State to this path. 🔴 This is what makes "
        "the WebGPU engine comparable with the numbers in this report: the "
        "force error has to be evaluated at the *same* configuration, and "
        "equilibration is stochastic, so re-running it elsewhere gives a "
        "different structure and an incomparable number.",
    )
    args = p.parse_args()

    state = XmlSerializer.deserialize(open(args.state_xml).read())
    positions = state.getPositions()
    box = state.getPeriodicBoxVectors()
    velocities = None
    equilibration = None

    report: dict = {
        "host": {"platform": platform.platform(), "processor": platform.processor()},
        "openmm": openmm.version.version,
        "fixture": args.system_xml,
        "available_platforms": [openmm.Platform.getPlatform(i).getName()
                                for i in range(openmm.Platform.getNumPlatforms())],
    }
    if args.equilibrate_ps > 0:
        positions, velocities, box, equilibration = equilibrate(
            args.system_xml, positions, box, args.equilibrate_ps)
        print(f"equilibrated {equilibration['picoseconds']} ps on "
              f"{equilibration['platform']} in {equilibration['seconds']} s -> "
              f"{equilibration['potential_energy_kJ_mol']:.1f} kJ/mol")
    report["equilibration"] = equilibration

    # the double-precision Reference answer is the baseline for everything
    made = make_context(args.system_xml, "Reference", None, grouped=True)
    assert made is not None, "the Reference platform is always present"
    ref_ctx, names = made
    report["force_groups"] = names
    ref = single_point(ref_ctx, names, positions, box)
    report["reference"] = {"energies": ref["energies"], "total": ref["total"],
                           "force_rms_kJ_mol_nm": float(
                               np.sqrt((ref["forces"] ** 2).sum(axis=1).mean()))}
    print("Reference total:", ref["total"], "kJ/mol")
    del ref_ctx

    # What the machine says it has, independent of what OpenMM managed to use.
    try:
        import subprocess
        report["host"]["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip() or "(no output)"
    except Exception as exc:
        report["host"]["nvidia_smi"] = f"unavailable: {exc}"
    print("  host GPU:", report["host"]["nvidia_smi"])

    report["platforms"] = {}
    for name, precision in CANDIDATES:
        label = f"{name}/{precision}" if precision else name
        made = make_context(args.system_xml, name, precision, grouped=True)
        if made is None:
            continue
        ctx, names2 = made
        entry = {}
        try:
            got = single_point(ctx, names2, positions, box)
            entry["device"] = ctx.getPlatform().getPropertyValue(ctx, "DeviceName") \
                if name in ("CUDA", "OpenCL", "HIP") else platform.processor()
            entry["single_point_vs_reference"] = compare(ref, got)
            print(f"  {label}: force rel RMS {entry['single_point_vs_reference']['force_rel_rms']:.3e}"
                  f"  total {entry['single_point_vs_reference']['total_abs_kJ_mol']:+.4f} kJ/mol")
        except Exception as exc:
            entry["single_point_vs_reference"] = f"failed: {exc}"
        del ctx
        if not args.skip_nve:
            entry["nve"] = nve_drift(args.system_xml, name, precision, positions, box,
                                     args.nve_steps, args.nve_report, velocities)
            if entry["nve"]:
                print(f"  {label}: NVE drift "
                      f"{entry['nve']['drift_kJ_mol_per_ns_per_atom']:+.3e} kJ/mol/ns/atom")
        if not args.skip_speed:
            entry["speed"] = speed(args.system_xml, name, precision, positions, box,
                                   args.speed_steps)
            if entry["speed"]:
                print(f"  {label}: {entry['speed']['ns_per_day']} ns/day")
        report["platforms"][label] = entry

    # the thresholds the engine is held to: mixed precision, if it was measured
    basis_order = ["CUDA/mixed", "HIP/mixed", "OpenCL/mixed", "OpenCL/single", "CUDA/single"]
    basis = next((b for b in basis_order if b in report["platforms"]), None)
    mixed = report["platforms"].get(basis) if basis else None
    if mixed and isinstance(mixed.get("single_point_vs_reference"), dict):
        sp = mixed["single_point_vs_reference"]
        report["proposed_thresholds"] = {
            "basis": basis,
            "basis_is_mixed_precision": basis.endswith("/mixed"),
            "device": mixed.get("device"),
            "force_rel_rms": sp["force_rel_rms"],
            "force_max_error_kJ_mol_nm": sp["force_max_error_kJ_mol_nm"],
            "total_energy_abs_kJ_mol": abs(sp["total_abs_kJ_mol"]),
            "nve_drift_kJ_mol_per_ns_per_atom":
                abs(mixed.get("nve", {}).get("drift_kJ_mol_per_ns_per_atom", float("nan")))
                if isinstance(mixed.get("nve"), dict) else None,
            "note": "the WebGPU engine must not exceed these by more than the factor "
                    "recorded in AGENTS.md; raising the factor is a reviewed change. "
                    "A single-precision basis is weaker than the intended "
                    "mixed-precision one and must be replaced by a CUDA/mixed run.",
        }
        print("\nproposed thresholds:",
              json.dumps(report["proposed_thresholds"], indent=1))
    else:
        print("\nNo mixed-precision GPU platform on this host, so no thresholds were "
              "proposed.  Run this on a CUDA machine (Colab T4).")
    report["skipped_platforms"] = dict(SKIPPED)
    gpu_rows = [k for k in report["platforms"] if k.split("/")[0] in ("CUDA", "HIP", "OpenCL")]
    report["verdict"] = {
        "gpu_row_present": bool(gpu_rows),
        "gpu_rows": gpu_rows,
        "thresholds_proposed": "proposed_thresholds" in report,
        "summary": (
            "usable: a GPU platform produced a row"
            if gpu_rows
            else "NOT USABLE for calibration: no GPU platform produced a row, so "
                 "the mixed-precision basis the thresholds need is missing. See "
                 "skipped_platforms for the reason each candidate failed."
        ),
    }
    if not gpu_rows:
        print("\n🔴 " + report["verdict"]["summary"])
        for label, reason in SKIPPED.items():
            print(f"     {label}: {reason}")
    if args.save_equilibrated:
        equilibrated = openmm.State()
        # There is no public State constructor, so the state is rebuilt through
        # a throwaway Reference context at the equilibrated coordinates.
        scratch = XmlSerializer.deserialize(open(args.system_xml).read())
        ctx = openmm.Context(
            scratch, openmm.VerletIntegrator(0.002),
            openmm.Platform.getPlatformByName("Reference"),
        )
        ctx.setPeriodicBoxVectors(*box)
        ctx.setPositions(positions)
        if velocities is not None:
            ctx.setVelocities(velocities)
        equilibrated = ctx.getState(getPositions=True, getVelocities=True)
        open(args.save_equilibrated, "w").write(XmlSerializer.serialize(equilibrated))
        print("wrote", args.save_equilibrated)
        del ctx

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
