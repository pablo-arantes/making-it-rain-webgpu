"""What `Modeller.addSolvent` produced: the box, the waters, the ion counts.

Split into two runs on purpose, because only part of `addSolvent` is
reproducible.

The water placement is fully deterministic: tile a pre-equilibrated box, keep
the molecules whose oxygen falls inside the unit cell, drop the ones clashing
with the solute by summed van der Waals radii, then drop the ones clashing
across the periodic edge.  Run once with ions switched off and every water is
comparable, position by position.

Ion *placement* is not.  `_addIons` picks the waters to replace with
`random.choice`, so which water becomes a sodium differs between runs of OpenMM
itself.  The ion *counts* are deterministic -- they follow from the solute
charge and the water count -- so the second run records those and nothing about
where they went.  See docs/DEVIATIONS.md.

    python tools/dump_solvent.py --pdb fixtures/prep/1L2Y.pdb \\
        --out fixtures/prep/1L2Y-solvent.json
"""

import argparse
import json
import pathlib

import openmm.app as app
import openmm.version
from openmm import unit
from openmm.app import element


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--forcefield", nargs="+",
                        default=["amber14/protein.ff14SB.xml", "amber14/tip3p.xml"])
    parser.add_argument("--water", default="tip3p")
    # The notebook's defaults: 4 angstrom of padding, 0.15 M NaCl.
    parser.add_argument("--padding", type=float, default=4.0, help="angstrom")
    parser.add_argument("--concentration", type=float, default=0.15, help="molar")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    pdb = app.PDBFile(str(args.pdb))
    field = app.ForceField(*args.forcefield)
    solute = pdb.topology.getNumAtoms()

    def solvate(ions: bool):
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addSolvent(
            field,
            model=args.water,
            padding=args.padding * unit.angstrom,
            ionicStrength=(args.concentration if ions else 0.0) * unit.molar,
            neutralize=ions,
            positiveIon="Na+",
            negativeIon="Cl-",
        )
        return modeller

    # --- Run 1: no ions, so the water set is reproducible -------------------
    plain = solvate(False)
    vectors = plain.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    positions = plain.positions.value_in_unit(unit.nanometer)
    oxygens = [
        list(positions[atom.index])
        for atom in plain.topology.atoms()
        if atom.residue.name == "HOH" and atom.element == element.oxygen
    ]

    # --- Run 2: the notebook's real settings, for the counts ---------------
    ionic = solvate(True)
    counts: dict[str, int] = {}
    for residue in ionic.topology.residues():
        if residue.name not in ("HOH",):
            if len(list(residue.atoms())) == 1:
                counts[residue.name] = counts.get(residue.name, 0) + 1
    waters = sum(1 for r in ionic.topology.residues() if r.name == "HOH")

    out = {
        "pdb": args.pdb.name,
        "openmm": openmm.version.version,
        "water": args.water,
        "padding_nm": args.padding / 10.0,
        "concentration_molar": args.concentration,
        "soluteAtoms": solute,
        "noIons": {
            "boxVectors": [list(v) for v in vectors],
            "waters": len(oxygens),
            "atoms": plain.topology.getNumAtoms(),
            # Sorted, because the order waters are emitted in follows the tiling
            # loop and a set comparison is the honest one for "same waters".
            "oxygens": sorted(oxygens),
        },
        "withIons": {
            "waters": waters,
            "ionCounts": counts,
            "atoms": ionic.topology.getNumAtoms(),
        },
    }
    args.out.write_text(json.dumps(out))
    print(f"  box {[round(v[i], 4) for i, v in enumerate(vectors)]} nm")
    print(f"  no ions: {len(oxygens)} waters, {plain.topology.getNumAtoms()} atoms")
    print(f"  with ions: {waters} waters, {counts}, {ionic.topology.getNumAtoms()} atoms")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
