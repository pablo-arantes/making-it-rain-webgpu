#!/usr/bin/env python
"""What the notebook's equilibration restraint mask actually selects.

Cell 23 of reference/Uni_Dock+MD.ipynb builds its position-restraint list with

    pt.select_atoms('!(:H*) & !(:HOH) & !(:NA) & !(:CL) & !(:K)', pt_topology)

under the comment "positional restraints for all heavy-atoms".  In a cpptraj
mask `:` selects RESIDUE names and `@` selects ATOM names, so `:H*` excludes
residues whose name starts with H -- HOH, and also HIS -- while leaving every
hydrogen of every other residue in the selection.  This script measures both
masks on a built system so the difference is a number rather than an argument.

    python tools/probe_restraint_mask.py fixtures/3HTB/system.pdb
"""
import json
import sys

import pytraj as pt

MASKS = {
    "notebook_as_written": "!(:H*) & !(:HOH) & !(:NA) & !(:CL) & !(:K)",
    "heavy_atoms_as_intended": "!(@H*) & !(:HOH) & !(:NA) & !(:CL) & !(:K)",
    "residues_named_H_star": ":H*",
    "atoms_named_H_star": "@H*",
}


def main() -> None:
    top = pt.load_topology(sys.argv[1])
    out = {"topology": str(top), "atoms": top.n_atoms, "masks": {}}
    for label, mask in MASKS.items():
        idx = pt.select_atoms(mask, top)
        hydrogens = sum(1 for i in idx if top.atom(int(i)).atomic_number == 1)
        residues = sorted({top.residue(top.atom(int(i)).resid).name for i in idx})
        out["masks"][label] = {
            "mask": mask,
            "n_selected": int(len(idx)),
            "n_hydrogens_selected": hydrogens,
            "residue_names_covered": residues,
        }
        print(f"{label:26s} {mask!r}\n  selected {len(idx):6d} atoms, "
              f"{hydrogens:6d} of them hydrogen; HIS covered: {'HIS' in residues}")
    path = sys.argv[2] if len(sys.argv) > 2 else None
    if path:
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
