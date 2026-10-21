"""The ProLIF interaction fingerprint, with the ligand's chemistry supplied.

🔴 Getting a *meaningful* reference out of ProLIF takes two things that are
easy to miss, and without either it reports almost nothing while appearing to
work.

**The protein needs bonds.** MDAnalysis does not infer them from a PDB, so the
protein reaches RDKit as a bag of disconnected atoms and no SMARTS pattern
matches. Every interaction that depends on chemistry silently disappears and
only `VdWContact` -- which needs nothing but elements and distances --
survives. `guess_bonds()` fixes it.

**The ligand needs bond orders.** A PDB has none, so aromaticity and valence
are unknown. The ligand is therefore built from an SDF template and given the
trajectory's coordinates per frame.

Measured on 3HTB with JZ4: from the PDB alone ProLIF finds **one** interaction
(a VdW contact with ALA99). With both fixes it finds **sixteen** across the
hydrophobic cavity -- LEU91, ALA99, VAL111, PHE114, LEU118, LEU121, LEU133 and
PHE153, which is the L99A pocket. A parity test against the first version would
have been meaningless.

    ~/miniforge3/envs/prolif/bin/python tools/dump_interactions.py \\
        --dcd fixtures/analysis-3HTB/trajectory.dcd \\
        --pdb fixtures/3HTB/system.pdb --sdf fixtures/3HTB/ligand.sdf \\
        --ligand UNK --out fixtures/analysis-3HTB/interactions.json
"""

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcd", required=True, type=pathlib.Path)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--sdf", required=True, type=pathlib.Path)
    parser.add_argument("--ligand", default="UNK")
    parser.add_argument("--frames", type=int, default=0, help="0 means all")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    import MDAnalysis as mda
    import prolif
    from rdkit import Chem

    u = mda.Universe(str(args.pdb), str(args.dcd))
    ligand_ag = u.select_atoms(f"resname {args.ligand}")
    protein_ag = u.select_atoms("protein")
    if len(ligand_ag) == 0:
        raise SystemExit(f"no atoms for resname {args.ligand}")
    protein_ag.guess_bonds()

    template = Chem.MolFromMolFile(str(args.sdf), removeHs=False, sanitize=True)
    if template is None:
        raise SystemExit(f"{args.sdf} did not parse")
    if template.GetNumAtoms() != len(ligand_ag):
        raise SystemExit(
            f"{args.sdf} has {template.GetNumAtoms()} atoms and the trajectory's "
            f"{args.ligand} has {len(ligand_ag)}; they must correspond atom for atom"
        )

    order = list(ligand_ag.indices)
    frames = args.frames or len(u.trajectory)

    def ligand_frames():
        mol = Chem.Mol(template)
        for _ in u.trajectory[:frames]:
            conformer = mol.GetConformer()
            for slot, index in enumerate(order):
                x, y, z = u.atoms.positions[index]
                conformer.SetAtomPosition(slot, (float(x), float(y), float(z)))
            yield prolif.Molecule.from_rdkit(mol)

    protein = prolif.Molecule.from_mda(protein_ag)
    fp = prolif.Fingerprint()
    fp.run_from_iterable(ligand_frames(), protein, progress=False)
    frame = fp.to_dataframe()

    # Per (residue, interaction): the frequency over frames, and the per-frame
    # booleans so a reimplementation can be compared frame by frame rather than
    # only on an average that could hide compensating errors.
    columns = {}
    for column in frame.columns:
        key = f"{column[1]}|{column[2]}"
        series = frame[column].tolist()
        columns[key] = {
            "frequency": float(sum(series) / len(series)),
            "frames": [bool(v) for v in series],
        }

    # The chemistry ProLIF perceived for the ligand, so a reimplementation can
    # be given the same starting point rather than having to derive it.
    ligand_mol = prolif.Molecule.from_rdkit(template)
    chemistry = []
    for atom in ligand_mol.GetAtoms():
        chemistry.append(
            {
                "index": atom.GetIdx(),
                "element": atom.GetSymbol(),
                "aromatic": bool(atom.GetIsAromatic()),
                "formalCharge": atom.GetFormalCharge(),
                "degree": atom.GetDegree(),
                "totalHs": atom.GetTotalNumHs(includeNeighbors=True),
                "neighbours": sorted(n.GetSymbol() for n in atom.GetNeighbors()),
            }
        )

    hydrophobic = prolif.Fingerprint(["Hydrophobic"]).interactions["Hydrophobic"]
    out = {
        "prolif": prolif.__version__,
        "mdanalysis": mda.__version__,
        "frames": frames,
        "ligand": args.ligand,
        "ligandAtomIndices": [int(i) for i in order],
        "ligandChemistry": chemistry,
        "parameters": {
            "hydrophobicSmarts": "[c,s,Br,I,S&H0&v2,$([D3,D4;#6])&!$([#6]~[#7,#8,#9])&!$([#6X4H0]);+0]",
            "hydrophobicDistance_nm": 0.45,
            "vdwTolerance_nm": 0.0,
        },
        "interactions": columns,
        # Which ligand atoms ProLIF considers hydrophobic, which is the part a
        # reimplementation without a SMARTS engine has to reproduce by rule.
        "ligandHydrophobicAtoms": sorted(
            int(m[0]) for m in ligand_mol.GetSubstructMatches(hydrophobic.lig_pattern)
        ),
    }
    args.out.write_text(json.dumps(out, indent=1))
    kinds = sorted({k.split("|")[1] for k in columns})
    print(f"  {frames} frames, {len(columns)} (residue, interaction) pairs")
    print(f"  kinds: {kinds}")
    print(f"  ligand hydrophobic atoms: {out['ligandHydrophobicAtoms']}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
