"""The ligand analyses as the notebook computes them, as the reference.

Cells 33 to 37 of the notebook, each with the tool the notebook uses:

  * **ligand RMSD** (cell 34) -- `MDAnalysis.analysis.rms.RMSD` with
    `select="resname UNK and not (name H*)"` against frame 0.  🔴 Worth
    knowing what that measures: MDAnalysis superimposes the *selection* on the
    reference before computing the deviation, so this is the ligand's internal
    conformational change and **not** its movement in the pocket.  A ligand
    that slid two nanometres while staying rigid reports ~0.
  * **pocket distances** (cell 36) -- every residue with any atom within 4 A of
    the ligand *in the first frame*, then the centre-of-mass distance between
    the ligand and each of those residues, per frame.
  * **named residues** (cell 37) -- the same for residue numbers the user types.
  * **the interaction fingerprint** (cell 29) -- ProLIF over the frames.

Needs one environment with both MDAnalysis and ProLIF.  On this machine that is
`~/miniforge3/envs/prolif`.

    ~/miniforge3/envs/prolif/bin/python tools/dump_ligand_analyses.py \\
        --dcd fixtures/analysis-3HTB/trajectory.dcd \\
        --pdb fixtures/3HTB/system.pdb --ligand UNK \\
        --out fixtures/analysis-3HTB/ligand.json
"""

import argparse
import json
import pathlib

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcd", required=True, type=pathlib.Path)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--ligand", default="UNK", help="ligand residue name")
    parser.add_argument("--cutoff", type=float, default=4.0, help="angstrom")
    parser.add_argument("--residues", default="78,84", help="cell 37's default")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    import MDAnalysis
    from MDAnalysis.analysis import rms

    u = MDAnalysis.Universe(str(args.pdb), str(args.dcd))
    select = f"resname {args.ligand} and not (name H*)"
    ligand = u.select_atoms(select)
    if len(ligand) == 0:
        raise SystemExit(f"no atoms matched '{select}'")

    # cell 34, exactly as the notebook calls it.
    analysis = rms.RMSD(u, select=select).run()
    # columns are [frame, time, rmsd]; angstrom -> nm
    rmsd = (np.asarray(analysis.results.rmsd)[:, 2] / 10.0).tolist()

    # cell 36: the pocket is fixed by the *first* frame and then held.
    u.trajectory[0]
    near = u.select_atoms(
        f"(around {args.cutoff} group lig) and protein", lig=ligand, updating=False
    )
    pocket = sorted({int(a.resid) for a in near})

    named = [int(v) for v in args.residues.split(",") if v.strip()]

    def com_distances(resids):
        """Ligand centre of mass to each residue's, per frame, in nm."""
        groups = {r: u.select_atoms(f"protein and resid {r}") for r in resids}
        out = {str(r): [] for r in resids}
        for _ in u.trajectory:
            centre = ligand.center_of_mass()
            for r, g in groups.items():
                out[str(r)].append(float(np.linalg.norm(g.center_of_mass() - centre)) / 10.0)
        return out

    pocket_distances = com_distances(pocket)
    named_distances = com_distances(named)

    # 🔴 The masses MDAnalysis guessed, and the ligand's own centre of mass per
    # frame. `center_of_mass` is mass-weighted, so a reimplementation that uses
    # a different mass table lands somewhere slightly different -- recording
    # both turns that from a guess into a check.
    u.trajectory[0]
    masses = {int(a.index): float(a.mass) for a in ligand}
    ligand_com = []
    for _ in u.trajectory:
        ligand_com.append([float(v) / 10.0 for v in ligand.center_of_mass()])

    result = {
        "mdanalysis": MDAnalysis.__version__,
        "ligandMasses": masses,
        "ligandCentreOfMass": ligand_com,
        "frames": int(len(u.trajectory)),
        "ligandName": args.ligand,
        "ligandHeavyAtoms": [int(a.index) for a in ligand],
        "ligandRmsd": rmsd,
        "pocketCutoff_nm": args.cutoff / 10.0,
        "pocketResidues": pocket,
        "pocketDistances": pocket_distances,
        "namedResidues": named,
        "namedDistances": named_distances,
    }

    # cell 29, the interaction fingerprint.
    try:
        import prolif

        u2 = MDAnalysis.Universe(str(args.pdb), str(args.dcd))
        lig_sel = u2.select_atoms(f"resname {args.ligand}")
        prot_sel = u2.select_atoms("protein")
        fp = prolif.Fingerprint()
        fp.run(u2.trajectory, lig_sel, prot_sel, progress=False)
        frame = fp.to_dataframe()
        # Frequency of each (residue, interaction) over the frames, which is
        # what LigNetwork's aggregate view thresholds on.
        frequency = {
            f"{col[1]}|{col[2]}": float(frame[col].mean()) for col in frame.columns
        }
        result["prolif"] = prolif.__version__
        result["interactionFrequency"] = frequency
    except Exception as exc:  # recorded, not swallowed
        result["prolif"] = f"unavailable: {type(exc).__name__}: {exc}"

    args.out.write_text(json.dumps(result, indent=1))
    print(f"  ligand {args.ligand}: {len(ligand)} heavy atoms, {result['frames']} frames")
    print(f"  RMSD {min(rmsd):.5f}..{max(rmsd):.5f} nm")
    print(f"  pocket at {args.cutoff} A: {len(pocket)} residues {pocket}")
    if isinstance(result.get("interactionFrequency"), dict):
        print(f"  {len(result['interactionFrequency'])} interactions: "
              f"{sorted(result['interactionFrequency'])[:4]}")
    else:
        print(f"  prolif: {result['prolif']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
