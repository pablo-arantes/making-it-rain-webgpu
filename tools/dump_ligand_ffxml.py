"""A ligand's parameters as an OpenMM ffxml, which is the browser's upload format.

Phase 4 needs the ligand typed before a protein-ligand system can be built, and
the two routes have very different costs.  SMIRNOFF typing in the browser needs
SMARTS matching, which means RDKit in WebAssembly.  **Uploading parameters**
needs nothing new: `openmmforcefields` already emits an ffxml holding a residue
template plus the types, bonded terms and nonbonded parameters, and
`src/prep/ffxml.ts` already reads exactly that.

So a user parameterises the ligand once -- here, in the notebook, or anywhere
with AmberTools -- and the browser merges the result with its protein and water
force fields.  Template matching is by **graph**, so it does not matter that
this file names the residue after the molecule's SMILES and its atoms after
Tripos types: the ligand in the PDB is matched by its bonds and elements.

    ~/miniforge3/envs/parametrizani_env/bin/python tools/dump_ligand_ffxml.py \\
        --sdf fixtures/3HTB/ligand.sdf --out fixtures/ligand/JZ4-gaff.xml
"""

import argparse
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", required=True, type=pathlib.Path)
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "gaff", "smirnoff"],
        help="auto tries GAFF, which the notebook defaults to, then SMIRNOFF",
    )
    parser.add_argument("--gaff-version", default="gaff-2.11")
    parser.add_argument("--smirnoff-version", default="openff-2.0.0")
    parser.add_argument(
        "--charges",
        default="openff-gnn-am1bcc-1.0.0.pt",
        help="NAGL model for partial charges, or 'none' to let the generator "
        "assign them. 🔴 NAGL rather than AM1-BCC by default, and not only "
        "because AmberTools' sqm is unavailable here: the browser cannot run "
        "sqm either, so NAGL is the charge method the engine will actually use "
        "(docs/DEVIATIONS.md entry 6). Using it on both sides makes the "
        "comparison measure the typing rather than the charge method.",
    )
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from openff.toolkit.topology import Molecule

    molecule = Molecule.from_file(str(args.sdf), allow_undefined_stereo=True)
    if isinstance(molecule, list):
        molecule = molecule[0]
    print(f"  {molecule.to_smiles(explicit_hydrogens=False)}, "
          f"{molecule.n_atoms} atoms, charge {molecule.total_charge}")

    if args.charges != "none":
        from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper

        NAGLToolkitWrapper().assign_partial_charges(molecule, args.charges)
        total = sum(float(q.m) for q in molecule.partial_charges)
        print(f"  charges from {args.charges}: total {total:+.6f} e")

    order = (
        ["gaff", "smirnoff"] if args.method == "auto" else [args.method]
    )
    errors = {}
    for method in order:
        try:
            if method == "gaff":
                from openmmforcefields.generators import GAFFTemplateGenerator

                generator = GAFFTemplateGenerator(
                    molecules=[molecule], forcefield=args.gaff_version
                )
            else:
                from openmmforcefields.generators import SMIRNOFFTemplateGenerator

                generator = SMIRNOFFTemplateGenerator(
                    molecules=[molecule], forcefield=args.smirnoff_version
                )
            ffxml = generator.generate_residue_template(molecule)
        except Exception as exc:
            # 🔴 Recorded per method rather than swallowed: GAFF needs
            # antechamber from AmberTools and SMIRNOFF needs a charge backend,
            # and "it did not work" is useless without which one was missing.
            errors[method] = f"{type(exc).__name__}: {exc}"
            print(f"  {method}: unavailable ({exc})")
            continue

        args.out.write_text(ffxml)
        print(f"  {method} ({getattr(generator, 'forcefield', '?')}): "
              f"{len(ffxml)} bytes -> {args.out}")
        return

    raise SystemExit(
        "no typing route worked:\n"
        + "\n".join(f"  {k}: {v}" for k, v in errors.items())
    )


if __name__ == "__main__":
    main()
