#!/usr/bin/env python
"""Is the notebook's system.pdb reproducible, and is the box big enough?

Two questions that decide how preparation can be validated at all.

1. `Modeller.addHydrogens` places any hydrogen whose position it cannot derive
   geometrically at a *random* position -- `random.random()`, from Python's
   unseeded global RNG (openmm/app/modeller.py:1026) -- and then relaxes the
   added hydrogens with 50 minimiser iterations at a 1.0 kJ/mol/nm tolerance
   (modeller.py:1043, 1103).  `Modeller.addSolvent` likewise picks which water
   becomes an ion with `random.choice` (modeller.py:355).  So repeating the
   preparation does not repeat the structure.  This measures by how much.

2. `addSolvent(padding=p)` sets the box to `max(2*radius + p, 2*p)` where
   `radius` is the bounding-SPHERE radius about the bounding-box centre
   (modeller.py:499-512).  The padding is therefore the TOTAL added width --
   p/2 per side -- and not "(solute size) + 2*padding" as the notebook's text
   says.  This measures the resulting minimum distance between a solute atom
   and its own periodic image, which is what actually matters against the
   nonbonded cutoff.

    python tools/probe_preparation_determinism.py receptor.pdb \
        --solvated fixtures/3HTB/system.pdb --trials 3 \
        --out docs/measurements/preparation-determinism.json
"""
import argparse
import json

import numpy as np
import openmm.app as app
from openmm import unit


def bounding_radius(positions: np.ndarray) -> float:
    centre = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
    return float(np.linalg.norm(positions - centre, axis=1).max())


def hydrogen_determinism(receptor: str, trials: int, ph: float, padding_nm: float) -> dict:
    from pdbfixer import PDBFixer

    runs = []
    for _ in range(trials):
        fixer = PDBFixer(filename=receptor)
        fixer.removeHeterogens()
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=ph)
        pos = np.array(fixer.positions.value_in_unit(unit.nanometer))
        runs.append({"atoms": fixer.topology.getNumAtoms(), "positions": pos})
    radii = [bounding_radius(r["positions"]) for r in runs]
    displacements = []
    for i in range(1, len(runs)):
        if runs[i]["positions"].shape == runs[0]["positions"].shape:
            displacements.append(
                float(np.linalg.norm(runs[i]["positions"] - runs[0]["positions"],
                                     axis=1).max()))
    return {
        "trials": trials,
        "atoms": [r["atoms"] for r in runs],
        "bounding_radius_nm": [round(r, 6) for r in radii],
        "box_at_padding_nm": [round(2 * r + padding_nm, 6) for r in radii],
        "max_per_atom_displacement_vs_trial0_angstrom":
            [round(d * 10, 4) for d in displacements],
    }


def min_image_distance(solvated: str) -> dict:
    from scipy.spatial import cKDTree

    pdb = app.PDBFile(solvated)
    solute = [a.index for a in pdb.topology.atoms()
              if a.residue.name not in ("HOH", "NA", "CL", "K")]
    pos = np.array(pdb.positions.value_in_unit(unit.nanometer))[solute]
    vectors = np.array(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    box = np.array([vectors[0][0], vectors[1][1], vectors[2][2]])
    best = np.inf
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                d, _ = cKDTree(pos + np.array([dx, dy, dz]) * box).query(pos, k=1)
                best = min(best, float(d.min()))
    return {
        "file": solvated, "solute_atoms": len(solute),
        "box_nm": round(float(box[0]), 6),
        "min_solute_to_image_nm": round(best, 6),
        "inside_1nm_cutoff": best < 1.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("receptor", nargs="?", help="PDB to run PDBFixer on")
    p.add_argument("--solvated", action="append", default=[],
                   help="a built system.pdb to measure the periodic image distance of")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--ph", type=float, default=7.4)
    p.add_argument("--padding", type=float, default=4.0, help="angstrom")
    p.add_argument("--out")
    args = p.parse_args()

    out = {}
    if args.receptor:
        out["addHydrogens_determinism"] = hydrogen_determinism(
            args.receptor, args.trials, args.ph, args.padding / 10.0)
        print(json.dumps(out["addHydrogens_determinism"], indent=1))
    if args.solvated:
        out["periodic_image"] = [min_image_distance(f) for f in args.solvated]
        for row in out["periodic_image"]:
            print(f"{row['file']}: box {row['box_nm']} nm, nearest own image "
                  f"{row['min_solute_to_image_nm']} nm"
                  + ("  <-- INSIDE the 1 nm cutoff" if row["inside_1nm_cutoff"] else ""))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
