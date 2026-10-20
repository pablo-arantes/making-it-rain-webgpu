"""Can OpenMM read a PDB this engine rewrote, with its topology intact?

Each stage writes a PDB of its final frame as the analysis topology, built by
substituting coordinates into the file the user uploaded.  Two things therefore
have to hold, and only the second needs OpenMM: the coordinates must come back
to the precision the format carries, and the *topology* -- names, residues,
chains, elements, bonds inferred from residue templates -- must be exactly what
the original file produced.

The second is the one worth checking.  A fixed-width format rewritten with a
column off by one still parses: the atom names shift, the residue templates stop
matching, and the file silently describes a different molecule.

    python tools/check_pdb.py --pdb fixtures/pdb-roundtrip/ours.pdb \\
        --original fixtures/1L2Y/system.pdb \\
        --expected fixtures/pdb-roundtrip/expected.json
"""

import argparse
import json
import pathlib

import numpy as np
from openmm import app, unit


def topology_signature(topology):
    """Everything about a topology that a coordinate rewrite must not change."""
    return {
        "atoms": [
            (
                atom.name,
                atom.residue.name,
                atom.residue.id,
                atom.residue.chain.id,
                None if atom.element is None else atom.element.symbol,
            )
            for atom in topology.atoms()
        ],
        "bonds": sorted(
            tuple(sorted((bond[0].index, bond[1].index))) for bond in topology.bonds()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--original", required=True, type=pathlib.Path)
    parser.add_argument("--expected", required=True, type=pathlib.Path)
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text())
    ours = app.PDBFile(str(args.pdb))
    theirs = app.PDBFile(str(args.original))

    mine = topology_signature(ours.topology)
    original = topology_signature(theirs.topology)
    positions = np.asarray(
        ours.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ).ravel()
    wanted = np.array(expected["positions"])
    box = ours.topology.getUnitCellDimensions()

    report = {
        "atoms": ours.topology.getNumAtoms(),
        "topology_identical": mine == original,
        "worst_coordinate_error_nm": float(np.abs(positions - wanted).max()),
        "tolerance_nm": expected["tolerance_nm"],
        "box_nm": None if box is None else [float(v) for v in box.value_in_unit(unit.nanometer)],
        "box_expected": expected["box"],
    }
    report["coordinates_within_format"] = (
        report["worst_coordinate_error_nm"] <= expected["tolerance_nm"]
    )
    report["box_matches"] = report["box_nm"] is not None and all(
        abs(a - b) < 1e-3 for a, b in zip(report["box_nm"], expected["box"])
    )
    report["ok"] = all(
        (
            report["topology_identical"],
            report["coordinates_within_format"],
            report["box_matches"],
            report["atoms"] == expected["atoms"],
        )
    )
    print(json.dumps(report, indent=1))
    if not report["topology_identical"]:
        for i, (a, b) in enumerate(zip(mine["atoms"], original["atoms"])):
            if a != b:
                print(f"  first atom difference at {i}: ours {a}, original {b}")
                break
    if not report["ok"]:
        raise SystemExit(1)
    print("  OpenMM read it back with the topology intact")


if __name__ == "__main__":
    main()
