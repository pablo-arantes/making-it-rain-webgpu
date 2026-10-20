"""What `Modeller.addHydrogens` produced: the atoms, the bonds, the variants.

Phase 3 reimplements hydrogen addition, and the comparable part of it is the
*topology*: which variant each residue takes, which hydrogens get added, what
they are called, and where they land in the atom order.

The positions are deliberately not compared, and the reason is in
`modeller.py`: each added hydrogen starts at its parent plus a 0.1 nm delta
built from `random.random()`, and the whole set is then energy-minimised. So
OpenMM does not reproduce its own hydrogen positions between runs unless the
interpreter's RNG is seeded. There is nothing there to match.  See
docs/DEVIATIONS.md.

Hydrogens are stripped first, so the comparison is against what OpenMM builds
from scratch rather than against whatever the depositor named their hydrogens.

    python tools/dump_hydrogens.py --pdb fixtures/1L2Y/receptor.pdb \\
        --out fixtures/prep/1L2Y-hydrogens.json
"""

import argparse
import json
import pathlib

import openmm.app as app
import openmm.version
from openmm.app import element


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--ph", type=float, default=7.0)
    parser.add_argument(
        "--forcefield",
        nargs="*",
        default=["amber14/protein.ff14SB.xml"],
        help="passed to addHydrogens, which uses it for the position "
        "minimisation only; pass none to use the generic restraint system",
    )
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    pdb = app.PDBFile(str(args.pdb))
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    modeller.delete(
        [a for a in modeller.topology.atoms() if a.element == element.hydrogen]
    )

    stripped = [
        [a.name, a.residue.name, a.residue.id, a.residue.chain.id]
        for a in modeller.topology.atoms()
    ]

    field = app.ForceField(*args.forcefield) if args.forcefield else None
    variants = modeller.addHydrogens(field, pH=args.ph)

    topology = modeller.topology
    out = {
        "pdb": args.pdb.name,
        "ph": args.ph,
        "openmm": openmm.version.version,
        "strippedAtoms": stripped,
        # One entry per residue, in topology order. `null` means the residue has
        # no hydrogen definitions at all (water, ions).
        "variants": variants,
        "residues": [
            [r.name, r.id, r.chain.id] for r in topology.residues()
        ],
        # The result: every atom in order, so both which hydrogens were added
        # and where they were inserted are checkable.
        "atoms": [
            [a.name, a.element.symbol if a.element is not None else None,
             a.residue.index]
            for a in topology.atoms()
        ],
        "bonds": sorted(
            [min(b[0].index, b[1].index), max(b[0].index, b[1].index)]
            for b in topology.bonds()
        ),
    }

    args.out.write_text(json.dumps(out, indent=1))
    added = sum(1 for a in out["atoms"] if a[1] == "H")
    print(f"  {len(stripped)} heavy atoms -> {len(out['atoms'])} atoms ({added} H)")
    chosen = sorted({v for v in variants if v is not None})
    print(f"  variants chosen: {' '.join(chosen) if chosen else '(none)'}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
