"""A protein with hydrogens and no solvent, for testing parameter assignment.

Phase 3's gate is term-by-term parameter parity, and the only way to measure it
is on a system where nothing else is happening.  Solvation adds thousands of
water molecules whose parameters are trivial and whose presence hides a single
wrong protein torsion in the noise of a diff; `addHydrogens` makes choices of
its own that belong to a different test.

So this prepares the smallest thing that still exercises the hard part: one
chain, standard residues, explicit hydrogens, no box, no constraints.  Every
bond, angle, torsion, charge and Lennard-Jones pair in the resulting System came
from the residue templates and the parameter tables, which is exactly what a
reimplementation has to reproduce.

    python tools/build_prep_fixture.py --pdb fixtures/1L2Y/receptor.pdb \\
        --out fixtures/prep/1L2Y
"""

import argparse
import pathlib

import openmm
from openmm import app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument(
        "--forcefield", nargs="+", default=["amber14/protein.ff14SB.xml"]
    )
    parser.add_argument("--ph", type=float, default=7.0)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pdb = app.PDBFile(str(args.pdb))
    field = app.ForceField(*args.forcefield)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # Whatever hydrogens the file carried are replaced, so the set is the force
    # field's own and not the depositor's -- an NMR structure's hydrogens may be
    # named differently from the templates.
    modeller.deleteWater()
    modeller.delete([a for a in modeller.topology.atoms() if a.element == app.element.hydrogen])
    modeller.addHydrogens(field, pH=args.ph)

    out = args.out.with_suffix(".pdb")
    with open(out, "w") as handle:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, handle, keepIds=True)

    residues = sorted({r.name for r in modeller.topology.residues()})
    print(
        f"  {modeller.topology.getNumAtoms()} atoms, "
        f"{modeller.topology.getNumResidues()} residues, "
        f"{len(list(modeller.topology.bonds()))} bonds"
    )
    print(f"  residues: {' '.join(residues)}")
    print(f"  wrote {out}")
    print(f"  openmm {openmm.version.version}")


if __name__ == "__main__":
    main()
