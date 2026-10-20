"""Per-force-group energies from OpenMM, for a System this project wrote.

The plan's Phase 3 gate.  Term-by-term parameter parity shows the two engines
assign the same numbers to the same terms; it does not show they agree on an
energy.  A wrong unit in the serialiser, or a force written with the wrong
convention, survives every structural comparison and appears only here.

Run on double precision Reference, so the number is OpenMM's best answer rather
than a platform's.  Each force is put in its own group first, because a total is
a sum: a bonded term too high and a nonbonded term too low cancel, and one
number cannot say which.

    python tools/cross_feed_energy.py --system fixtures/crossfeed/system.xml \\
        --state fixtures/crossfeed/state.xml --out fixtures/crossfeed/energies.json
"""

import argparse
import json
import pathlib

import openmm
import openmm.version
from openmm import unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, type=pathlib.Path)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    system = openmm.XmlSerializer.deserialize(args.system.read_text())
    state = openmm.XmlSerializer.deserialize(args.state.read_text())

    # One group per force, so a disagreement names the force.
    names = []
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
        names.append(type(force).__name__)

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(state.getPositions())
    context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())

    groups = {}
    for index, name in enumerate(names):
        energy = context.getState(getEnergy=True, groups={index}).getPotentialEnergy()
        groups[f"{index}:{name}"] = energy.value_in_unit(unit.kilojoule_per_mole)
    total = context.getState(getEnergy=True).getPotentialEnergy()

    # 🔴 The PME mesh and splitting parameter OpenMM actually used. The
    # Reference platform takes the exact minimum grid rather than rounding up
    # to a smooth FFT size, so a reciprocal-space comparison that assumes the
    # smooth size is comparing two different meshes and will disagree by far
    # more than round-off. Asked for rather than recomputed.
    pme = None
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        if isinstance(force, openmm.NonbondedForce):
            alpha, nx, ny, nz = force.getPMEParametersInContext(context)
            pme = {
                "alpha_per_nm": alpha.value_in_unit(unit.nanometer**-1)
                if unit.is_quantity(alpha)
                else float(alpha),
                "grid": [int(nx), int(ny), int(nz)],
            }
            break

    forces = context.getState(getForces=True).getForces(asNumpy=True)
    out = {
        "openmm": openmm.version.version,
        "platform": "Reference",
        "particles": system.getNumParticles(),
        "constraints": system.getNumConstraints(),
        "byForce": groups,
        "pme": pme,
        "total": total.value_in_unit(unit.kilojoule_per_mole),
        # A few forces, so a comparison can check the gradient and not only the
        # energy -- an energy can be right with the wrong sign convention
        # somewhere downstream of it.
        "forceSample": [
            [float(v) for v in forces[i].value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer)]
            for i in (0, 1, 2, 100, 1000)
        ],
    }
    args.out.write_text(json.dumps(out, indent=1))
    for key, value in groups.items():
        print(f"  {key:<34} {value:16.6f} kJ/mol")
    print(f"  {'TOTAL':<34} {out['total']:16.6f} kJ/mol")
    if pme is not None:
        print(f"  PME alpha {pme['alpha_per_nm']:.6f} /nm, grid {pme['grid']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
