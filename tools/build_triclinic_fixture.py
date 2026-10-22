"""A triclinic fixture: the oracle the reference's triclinic path is tested on.

    ~/miniforge3/envs/parametrizani_env/bin/python -I tools/build_triclinic_fixture.py \
        --shape dodecahedron --out fixtures/1L2Y-dodecahedron

🔴 **Use the oracle environment, which is OpenMM 8.4 or newer.** OpenMM 7.7's
`addSolvent(boxShape=...)` is broken: it removes clashes across a *rectangular*
boundary, so a dodecahedron comes out with atoms 0.0168 nm apart and a potential
energy of 9.6e10 kJ/mol. The cube is unaffected, so nothing warns you -- the
fixture simply encodes a structure no force field can describe, and every
differential test against it then compares two engines' handling of an
absurdity. On 8.4 the same call gives 0.0636 nm and -7,633 kJ/mol.
`docs/measurements/README.md` names the environment; `tools/environment.yml` is
what it should be built from.

This deliberately does not use `build_reference_system.py`: that one needs
pdbfixer, which the oracle environment does not have, and the committed
`fixtures/1L2Y/1L2Y.pdb` has already been through it. Starting from that file
means this tool adds solvent and nothing else.

Dumps a minimised state as well as the raw one. The raw solvated structure has
contacts tight enough that Lennard-Jones dominates every force, which makes a
relative force comparison report on the structure rather than on the
implementation -- the same reason the acceptance thresholds are defined at an
equilibrated frame.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openmm import XmlSerializer

WATER = "amber14/tip3p.xml"
PROTEIN = "amber14/protein.ff14SB.xml"


def closest_approach(positions: np.ndarray, box: np.ndarray, residues: np.ndarray,
                     sample: int = 1500) -> float:
    """Closest inter-residue approach under the triclinic minimum image."""
    a, b, c = box
    chosen = np.random.default_rng(0).choice(
        len(positions), size=min(sample, len(positions)), replace=False
    )
    worst = float("inf")
    for i in chosen:
        d = positions - positions[i]
        d = d - np.outer(np.round(d[:, 2] / c[2]), c)
        d = d - np.outer(np.round(d[:, 1] / b[1]), b)
        d = d - np.outer(np.round(d[:, 0] / a[0]), a)
        r = np.linalg.norm(d, axis=1)
        r[residues == residues[i]] = np.inf
        worst = min(worst, float(r.min()))
    return worst


def platform_energies(
    system: openmm.System,
    state: openmm.State,
    names: list[str],
    platform: str,
) -> dict[str, float]:
    """Per-group energies on one platform, from a serialised state."""
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.002),
        openmm.Platform.getPlatformByName(platform),
    )
    context.setState(state)
    out = group_energies(context, names)
    del context
    return out


def missing_pairs(box: np.ndarray, cutoff: float, samples: int = 4000) -> dict:
    """How many in-cutoff pairs OpenMM's Reference platform fails to report.

    🔴 This is why the fixture records two platforms. For a triclinic cell the
    Reference platform's neighbour list misses pairs just inside the cutoff --
    measured below -- so it cannot be the oracle for triclinic direct space,
    although it is the oracle everywhere else in this project. The CPU platform
    agrees with an exact lattice search.
    """
    import itertools

    a, b, c = box
    system = openmm.System()
    for _ in range(2):
        system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box])
    force = openmm.CustomNonbondedForce("r")
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff)
    for _ in range(2):
        force.addParticle([])
    system.addForce(force)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    rng = np.random.default_rng(13)
    inside = 0
    missed = 0
    closest_missed = cutoff
    for _ in range(samples):
        d = rng.uniform(-4, 4, 3)
        true = min(
            float(np.linalg.norm(d + i * a + j * b + k * c))
            for i, j, k in itertools.product(range(-3, 4), repeat=3)
        )
        if true >= cutoff * 0.999:
            continue
        inside += 1
        context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(*d)])
        reported = (
            context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        if reported == 0.0:
            missed += 1
            closest_missed = min(closest_missed, true)
    del context
    return {
        "sampled_inside_cutoff": inside,
        "missed_by_reference_platform": missed,
        "closest_missed_nm": closest_missed,
        "note": "Pairs whose true nearest-image separation is inside the cutoff "
                "that the Reference platform reports as non-interacting. Zero on "
                "a cube.",
    }


def group_energies(context: openmm.Context, names: list[str]) -> dict[str, float]:
    out = {}
    for group, name in enumerate(names):
        out[f"{group}:{name}"] = (
            context.getState(getEnergy=True, groups={group})
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
    out["total"] = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", default="fixtures/1L2Y/1L2Y.pdb")
    parser.add_argument("--shape", default="dodecahedron",
                        choices=["cube", "dodecahedron", "octahedron"])
    parser.add_argument("--padding", type=float, default=0.4, help="nm")
    parser.add_argument("--cutoff", type=float, default=1.0, help="nm")
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimise-steps", type=int, default=2000)
    args = parser.parse_args()

    if openmm.version.version < "8":
        raise SystemExit(
            f"openmm {openmm.version.version} is too old: addSolvent(boxShape=...) "
            "removes clashes across a rectangular boundary before 8.x and "
            "produces an unusable structure. See this file's docstring."
        )

    os.makedirs(args.out, exist_ok=True)
    pdb = app.PDBFile(args.pdb)
    field = app.ForceField(PROTEIN, WATER)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(field, pH=7.4)
    modeller.addSolvent(
        field, model="tip3p", padding=args.padding * unit.nanometer,
        boxShape=args.shape, ionicStrength=0.15 * unit.molar,
        positiveIon="Na+", negativeIon="Cl-",
    )
    topology = modeller.getTopology()
    system = field.createSystem(
        topology, nonbondedMethod=app.PME,
        nonbondedCutoff=args.cutoff * unit.nanometer,
        constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005,
    )

    names = []
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)
        names.append(force.__class__.__name__)
    nonbonded = next(
        f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)
    )
    nonbonded.setReciprocalSpaceForceGroup(len(names))
    names.append("NonbondedForce[reciprocal]")

    context = openmm.Context(system, openmm.VerletIntegrator(0.002),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(modeller.getPositions())

    box = np.array(topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    residues = np.array([a.residue.index for a in topology.atoms()])
    raw = np.array(modeller.getPositions().value_in_unit(unit.nanometer))

    facts: dict = {
        "what": "A triclinic fixture for the reference's triclinic path, with "
                "OpenMM's per-force-group energies as the oracle.",
        "produced_by": "tools/build_triclinic_fixture.py",
        "openmm": openmm.version.version,
        "shape": args.shape,
        "padding_nm": args.padding,
        "cutoff_nm": args.cutoff,
        "atoms_total": topology.getNumAtoms(),
        "waters": sum(1 for r in topology.residues() if r.name == "HOH"),
        "box_nm": box.tolist(),
        "volume_nm3": float(abs(np.linalg.det(box))),
        "constraints": system.getNumConstraints(),
        "force_groups": names,
        "raw": {
            "energies_kJ_mol": group_energies(context, names),
            "closest_inter_residue_nm": closest_approach(raw, box, residues),
        },
    }

    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=args.minimise_steps)
    state = context.getState(getPositions=True, getVelocities=True, getForces=True)
    state_for_platforms = state
    minimised = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    facts["minimised"] = {
        "steps": args.minimise_steps,
        "energies_kJ_mol": group_energies(context, names),
        "energies_by_platform_kJ_mol": {
            platform: platform_energies(system, state_for_platforms, names, platform)
            for platform in ("Reference", "CPU")
        },
        "oracle": "CPU",
        "oracle_note":
            "🔴 Use the **CPU** platform for triclinic direct space, not "
            "Reference. The Reference platform's neighbour list misses pairs "
            "just inside the cutoff for a skewed cell -- see "
            "`reference_platform_pair_loss` -- which shows up as a few kJ/mol "
            "in the direct space and partially cancels between Coulomb and "
            "Lennard-Jones, so it reads as a small implementation difference "
            "rather than as missing pairs. Everywhere else in this project the "
            "Reference platform is the oracle, which is exactly why this is "
            "worth recording.",
        "closest_inter_residue_nm": closest_approach(
            np.asarray(minimised), box, residues
        ),
        "force_rms_kJ_mol_nm": float((forces ** 2).sum(axis=1).mean() ** 0.5),
        "force_max_component": float(abs(forces).max()),
        "note": "state.xml holds these coordinates. The raw block above is kept "
                "so the effect of minimising is visible rather than implied.",
    }

    facts["reference_platform_pair_loss"] = missing_pairs(box, args.cutoff)

    with open(os.path.join(args.out, "system.xml"), "w") as handle:
        handle.write(XmlSerializer.serialize(system))
    with open(os.path.join(args.out, "state.xml"), "w") as handle:
        handle.write(XmlSerializer.serialize(state))
    with open(os.path.join(args.out, "facts.json"), "w") as handle:
        json.dump(facts, handle, indent=1)
    # The run page needs a topology to name atoms in the DCD and the output
    # PDB, and it has to carry the *minimised* coordinates so it matches
    # state.xml rather than the structure before relaxation.
    with open(os.path.join(args.out, "system.pdb"), "w") as handle:
        app.PDBFile.writeFile(topology, state.getPositions(), handle)

    print(f"{args.shape}: {facts['atoms_total']} atoms, {facts['waters']} waters, "
          f"volume {facts['volume_nm3']:.2f} nm^3")
    print(f"  raw       total {facts['raw']['energies_kJ_mol']['total']:14.2f} kJ/mol, "
          f"closest {facts['raw']['closest_inter_residue_nm']:.4f} nm")
    loss = facts["reference_platform_pair_loss"]
    print(f"  Reference platform missed {loss['missed_by_reference_platform']} of "
          f"{loss['sampled_inside_cutoff']} in-cutoff pairs")
    print(f"  minimised total {facts['minimised']['energies_kJ_mol']['total']:14.2f} kJ/mol, "
          f"closest {facts['minimised']['closest_inter_residue_nm']:.4f} nm, "
          f"force RMS {facts['minimised']['force_rms_kJ_mol_nm']:.2f}")


if __name__ == "__main__":
    main()
