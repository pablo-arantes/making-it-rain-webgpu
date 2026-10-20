"""Pull one ligand out of a PDB as its own fragment, for the RDKit reference.

`dump_ligand.py` needs the ligand alone: `Chem.MolFromPDBFile` on a whole
structure gives the protein too.  This writes the HETATM records of a single
residue, which is what `build_reference_system.py` does internally.

    python tools/extract_ligand.py --pdb fixtures/ligand/181L.pdb --code BNZ \\
        --out fixtures/ligand/BNZ_fragment.pdb
"""

import argparse
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--code", required=True)
    parser.add_argument("--chain", help="restrict to one chain")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    code = args.code.upper()
    kept, seen = [], set()
    for line in args.pdb.read_text().splitlines():
        if not line.startswith("HETATM"):
            continue
        if line[17:20].strip().upper() != code:
            continue
        if args.chain is not None and line[21] != args.chain:
            continue
        key = (line[21], line[22:27])
        # 🔴 One copy only. A structure with the ligand bound twice would
        # otherwise produce a "molecule" of two disconnected fragments, which
        # perceives and types as neither.
        if seen and key not in seen:
            continue
        seen.add(key)
        kept.append(line)
    if not kept:
        raise SystemExit(f"no HETATM records for {code} in {args.pdb}")
    args.out.write_text("\n".join(kept) + "\nEND\n")
    print(f"  {len(kept)} atoms of {code} -> {args.out}")


if __name__ == "__main__":
    main()
