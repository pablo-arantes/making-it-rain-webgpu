"""What RDKit perceives for a ligand, as the reference for Phase 4's front end.

Typing a ligand needs its bond orders, and a PDB does not carry them: `ATOM`
records have elements and coordinates and nothing else.  The Python route gets
them from the RCSB Chemical Component Dictionary and transfers them onto the
fragment with `AssignBondOrdersFromTemplate`, which is a substructure match.

The browser cannot run RDKit's matcher, but it does not need to: the fragment
and the CCD component share **atom names**, so the orders can be transferred by
name.  That is only safe if it gives the same answer, which is what this dumps
so a test can check.

Reported per bond as a pair of atom *names*, which is what makes the two
routes comparable at all -- RDKit works in atom indices that depend on file
order.

    python tools/dump_ligand.py --cif /path/to/JZ4.cif \\
        --fragment fixtures/3HTB/ligand_fragment.pdb \\
        --ideal fixtures/3HTB/JZ4_ideal.sdf --out fixtures/ligand/JZ4.json
"""

import argparse
import json
import pathlib

from rdkit import Chem
from rdkit.Chem import AllChem

ORDER_NAMES = {
    Chem.BondType.SINGLE: "SING",
    Chem.BondType.DOUBLE: "DOUB",
    Chem.BondType.TRIPLE: "TRIP",
    Chem.BondType.AROMATIC: "AROM",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif", required=True, type=pathlib.Path)
    parser.add_argument("--fragment", required=True, type=pathlib.Path)
    parser.add_argument("--ideal", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    template = Chem.MolFromMolFile(str(args.ideal), removeHs=True, sanitize=True)
    if template is None:
        raise SystemExit(f"{args.ideal} did not parse")
    fragment = Chem.MolFromPDBFile(str(args.fragment), removeHs=True, sanitize=False)
    assigned = AllChem.AssignBondOrdersFromTemplate(template, fragment)
    Chem.SanitizeMol(assigned)

    # Atom names come from the PDB fragment's monomer info; the template's do
    # not survive an SDF, which is exactly why the comparison is by name on the
    # fragment side.
    names = []
    for atom in assigned.GetAtoms():
        info = atom.GetMonomerInfo()
        names.append(info.GetName().strip() if info is not None else atom.GetSymbol())

    bonds = []
    for bond in assigned.GetBonds():
        a = names[bond.GetBeginAtomIdx()]
        b = names[bond.GetEndAtomIdx()]
        low, high = sorted((a, b))
        bonds.append(
            {
                "atoms": [low, high],
                "order": ORDER_NAMES.get(bond.GetBondType(), str(bond.GetBondType())),
                "aromatic": bool(bond.GetIsAromatic()),
            }
        )
    bonds.sort(key=lambda row: (row["atoms"][0], row["atoms"][1]))

    withH = Chem.AddHs(assigned, addCoords=True)
    out = {
        "cif": args.cif.name,
        "rdkit": Chem.rdBase.rdkitVersion,
        "heavyAtomNames": names,
        "smiles": Chem.MolToSmiles(Chem.RemoveHs(assigned)),
        "templateSmiles": Chem.MolToSmiles(template),
        "formalCharge": Chem.GetFormalCharge(assigned),
        "heavyAtoms": assigned.GetNumAtoms(),
        "atomsWithHydrogens": withH.GetNumAtoms(),
        "bonds": bonds,
        "aromaticAtoms": sorted(
            names[a.GetIdx()] for a in assigned.GetAtoms() if a.GetIsAromatic()
        ),
        "formalCharges": {
            names[a.GetIdx()]: a.GetFormalCharge()
            for a in assigned.GetAtoms()
            if a.GetFormalCharge() != 0
        },
        "hydrogenCounts": {
            names[a.GetIdx()]: a.GetTotalNumHs() for a in assigned.GetAtoms()
        },
    }
    args.out.write_text(json.dumps(out, indent=1))
    print(f"  {out['smiles']}  ({out['heavyAtoms']} heavy, "
          f"{out['atomsWithHydrogens']} with H, charge {out['formalCharge']})")
    print(f"  {len(bonds)} bonds, {len(out['aromaticAtoms'])} aromatic atoms")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
