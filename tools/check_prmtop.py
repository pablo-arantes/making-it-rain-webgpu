"""Read a prmtop this project wrote, and compare its energy against the System.

The only verification that means anything for this format.  A prmtop with a
wrong exclusion window, a double-counted 1-4, or a Lennard-Jones type matrix off
by one loads perfectly and gives a plausible number -- so the test is not
"does it parse" but "does it produce the same energy as the System it came
from", per force group.

    python tools/check_prmtop.py --prmtop fixtures/amber/1L2Y.prmtop \\
        --inpcrd fixtures/amber/1L2Y.inpcrd \\
        --system fixtures/prep/1L2Y-system.xml --out fixtures/amber/1L2Y-energies.json
"""

import argparse
import json
import pathlib

import openmm
import openmm.app as app
from openmm import unit


def by_group(system, positions, boxVectors=None):
    names = []
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)
        names.append(type(force).__name__)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001 * unit.picoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(positions)
    if boxVectors is not None:
        context.setPeriodicBoxVectors(*boxVectors)
    out = {}
    for index, name in enumerate(names):
        value = context.getState(getEnergy=True, groups={index}).getPotentialEnergy()
        out[name] = out.get(name, 0.0) + value.value_in_unit(unit.kilojoule_per_mole)
    total = context.getState(getEnergy=True).getPotentialEnergy()
    return out, total.value_in_unit(unit.kilojoule_per_mole)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prmtop", required=True, type=pathlib.Path)
    parser.add_argument("--inpcrd", required=True, type=pathlib.Path)
    parser.add_argument("--system", required=True, type=pathlib.Path)
    parser.add_argument("--state", type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    prmtop = app.AmberPrmtopFile(str(args.prmtop))
    inpcrd = app.AmberInpcrdFile(str(args.inpcrd))
    periodic = prmtop.topology.getPeriodicBoxVectors() is not None

    # The same nonbonded treatment on both sides, or the comparison is between
    # two different physics rather than two files.
    method = app.PME if periodic else app.NoCutoff
    amber = prmtop.createSystem(
        nonbondedMethod=method,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )
    boxVectors = inpcrd.boxVectors if periodic else None
    amber_groups, amber_total = by_group(amber, inpcrd.positions, boxVectors)

    reference = openmm.XmlSerializer.deserialize(args.system.read_text())
    if args.state is not None:
        state = openmm.XmlSerializer.deserialize(args.state.read_text())
        positions = state.getPositions()
        refBox = state.getPeriodicBoxVectors() if periodic else None
    else:
        positions = inpcrd.positions
        refBox = boxVectors
    ref_groups, ref_total = by_group(reference, positions, refBox)

    out = {
        "prmtop": args.prmtop.name,
        "particles": amber.getNumParticles(),
        "periodic": bool(periodic),
        "fromPrmtop": amber_groups,
        "fromSystem": ref_groups,
        "totalFromPrmtop": amber_total,
        "totalFromSystem": ref_total,
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(f"  {amber.getNumParticles()} atoms, periodic={periodic}")
    for name in sorted(set(amber_groups) | set(ref_groups)):
        a = amber_groups.get(name, 0.0)
        b = ref_groups.get(name, 0.0)
        rel = abs(a - b) / max(1.0, abs(b))
        print(f"  {name:<22} prmtop {a:16.6f}  system {b:16.6f}  rel {rel:.3e}")
    rel = abs(amber_total - ref_total) / max(1.0, abs(ref_total))
    print(f"  {'TOTAL':<22} prmtop {amber_total:16.6f}  system {ref_total:16.6f}  rel {rel:.3e}")


if __name__ == "__main__":
    main()
