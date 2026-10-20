"""Which template OpenMM matched each residue to, and the atom correspondence.

Phase 3's template matcher is graph matching, and a graph can have more than one
valid isomorphism: tyrosine's phenol ring has a two-fold symmetry, so CD1/CD2
and CE1/CE2 can be mapped either way round with *identical* charges and types.
Comparing charges therefore cannot tell the two mappings apart.

It still matters which one you pick.  `ordering="amber"` breaks improper torsion
ties on `atomTemplateIndexes` -- the atom's index inside the template it matched
-- so a different-but-valid isomorphism reorders an improper's atoms, which
flips the sign of its dihedral angle and changes the energy.

OpenMM's matcher is compiled Cython, so the only way to know what it chose is to
ask it.  This dumps the template name per residue and the template atom index
per atom, which is exactly the quantity the ordering rules consume.

    python tools/dump_template_match.py --pdb fixtures/prep/1L2Y.pdb \\
        --forcefield amber14/protein.ff14SB.xml --out fixtures/prep/1L2Y-match.json
"""

import argparse
import json
import pathlib

import openmm.app as app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--forcefield", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    pdb = app.PDBFile(args.pdb)
    field = app.ForceField(*args.forcefield)

    # `_matchAllResiduesToTemplates` is what `createSystem` calls; going through
    # it directly is what gives access to the correspondence, which the public
    # API does not expose.
    data = app.ForceField._SystemData(pdb.topology)
    templates = field._matchAllResiduesToTemplates(data, pdb.topology, dict(), False)

    residues = list(pdb.topology.residues())
    out = {
        "pdb": pathlib.Path(args.pdb).name,
        "forcefield": list(args.forcefield),
        "openmm": openmm.version.version,
        "residues": [
            {"index": r.index, "name": r.name, "template": templates[r.index].name}
            for r in residues
        ],
        # One entry per atom, in topology order: the index of the template atom
        # it matched, and the type that came with it.
        "templateIndexes": [
            data.atomTemplateIndexes[atom] for atom in pdb.topology.atoms()
        ],
        "types": [data.atomType[atom] for atom in pdb.topology.atoms()],
        # 🔴 Neighbour lists in OpenMM's own order, which is bond-insertion
        # order and *not* sorted.  It reaches the energy: improper candidates
        # are `itertools.combinations` of this list, and the improper match
        # cache is keyed by type signature, so which candidate is seen first
        # decides the atom order every later candidate with that signature
        # replays.
        "bondedToAtom": [
            list(neighbours)
            for neighbours in field._buildBondedToAtomList(pdb.topology)
        ],
        "bonds": [[b[0].index, b[1].index] for b in pdb.topology.bonds()],
    }

    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"{len(residues)} residues, {len(out['templateIndexes'])} atoms -> {args.out}")


if __name__ == "__main__":
    import openmm.version

    main()
