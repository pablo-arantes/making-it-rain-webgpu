"""Add the notebook's position restraints to a fixture, and record them.

The restraint is added by the *run*, not by the force field, so no fixture
carries one until something puts it there.  This takes a built fixture, adds a
`CustomExternalForce` holding exactly the notebook's expression, and writes a
second system plus the restraint force group's own energy and forces.

The mask is the engine's default -- heavy atoms of everything that is not water
or an ion -- which is deviation 1 in docs/DEVIATIONS.md: the notebook's own
`!(:H*)` mask is a cpptraj slip that restrains 1318 hydrogens and leaves
histidine free, while its comments, its protocol and the papers all say heavy
atoms.  Both masks will be selectable; this fixture uses the default, and what
matters for a force comparison is only that the set is reproducible.

Putting the restraint in its own force group is what makes it separately
measurable: `getState(groups={N})` then reports the restraint alone, with no
need to subtract two large numbers that nearly cancel.

    python tools/build_restraint_fixture.py --fixture fixtures/1L2Y
"""

import argparse
import json
import pathlib

import numpy as np
import openmm
from openmm import app, unit

# Residue names that are solvent or counter-ions rather than solute.
SOLVENT = {"HOH", "WAT", "TIP3", "SOL", "NA", "CL", "K", "MG", "ZN", "CA"}

RESTRAINT_GROUP = 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=pathlib.Path)
    parser.add_argument(
        "--k",
        type=float,
        default=700.0,
        help=(
            "kJ/mol/nm^2. The notebook's `Force_constant` slider defaults to "
            "700 and its comment labels it 'kJ/mol' -- which is the wrong unit, "
            "since k multiplies a squared distance. One more behaviour-vs-label "
            "slip of the kind docs/NOTEBOOK_PARITY.md collects"
        ),
    )
    parser.add_argument("--platform", default="Reference")
    args = parser.parse_args()

    system = openmm.XmlSerializer.deserialize(
        (args.fixture / "system.xml").read_text()
    )
    state = openmm.XmlSerializer.deserialize((args.fixture / "state.xml").read_text())
    pdb = app.PDBFile(str(args.fixture / "system.pdb"))
    positions = np.array(
        state.getPositions().value_in_unit(unit.nanometer), dtype=float
    )

    restraint = openmm.CustomExternalForce(
        "k*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    for name in ("k", "x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    restraint.setForceGroup(RESTRAINT_GROUP)

    restrained = []
    for atom in pdb.topology.atoms():
        if atom.residue.name.upper() in SOLVENT:
            continue
        if atom.element is not None and atom.element.symbol == "H":
            continue
        index = atom.index
        restraint.addParticle(
            index,
            [
                args.k,
                positions[index][0],
                positions[index][1],
                positions[index][2],
            ],
        )
        restrained.append(index)

    if not restrained:
        raise SystemExit("the mask selected no atoms; check system.pdb's residues")
    system.addForce(restraint)

    out = args.fixture / "system-restrained.xml"
    out.write_text(openmm.XmlSerializer.serialize(system))

    # The restraint's own energy and forces, from its own group.  Displaced
    # coordinates as well as the reference ones: at the reference positions the
    # energy is exactly zero and every force vanishes, which would pass an
    # implementation that computed nothing at all.
    integrator = openmm.VerletIntegrator(0.002 * unit.picosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName(args.platform)
    )

    rng = np.random.default_rng(0x5EED)
    report = {
        "fixture": args.fixture.name,
        "openmm": openmm.version.version,
        "platform": args.platform,
        "k_kJ_mol_nm2": args.k,
        "force_group": RESTRAINT_GROUP,
        "restrained_atoms": len(restrained),
        "particles": system.getNumParticles(),
        "mask": "heavy atoms of non-solvent residues (the engine's default)",
        "cases": {},
    }

    for label, displacement in (("reference", 0.0), ("displaced", 0.05), ("wrapped", 3.0)):
        # The 3 nm case deliberately pushes atoms more than half a box, which
        # is what exercises the minimum image.  Without the wrap the energy
        # there is wrong by orders of magnitude; with the reference-position
        # case alone the test would pass on an implementation that returns 0.
        shifted = positions + displacement * rng.standard_normal(positions.shape)
        context.setPositions(shifted * unit.nanometer)
        measured = context.getState(
            getEnergy=True, getForces=True, groups={RESTRAINT_GROUP}
        )
        forces = measured.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        report["cases"][label] = {
            "displacement_nm": displacement,
            "energy_kJ_mol": measured.getPotentialEnergy().value_in_unit(
                unit.kilojoule_per_mole
            ),
            "force_rms": float(np.sqrt((forces ** 2).mean())),
            "max_force_component": float(np.abs(forces).max()),
        }
        np.save(args.fixture / f"restraint-{label}-positions.npy", shifted.ravel())
        np.save(args.fixture / f"restraint-{label}-forces.npy", forces.ravel())
        case = report["cases"][label]
        print(
            f"  {label:>10}  displaced {displacement:4.2f} nm  "
            f"energy {case['energy_kJ_mol']:14.4f}  "
            f"force rms {case['force_rms']:10.4f}  "
            f"max |f| {case['max_force_component']:12.4f}"
        )

    (args.fixture / "restraint-facts.json").write_text(json.dumps(report, indent=1))
    print(f"  {len(restrained)} atoms restrained; wrote {out.name} and restraint-facts.json")


if __name__ == "__main__":
    main()
