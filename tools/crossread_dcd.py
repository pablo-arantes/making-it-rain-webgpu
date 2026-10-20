"""Open a DCD with every available reader and check they agree with each other.

`check_dcd.py` compares a trajectory against the arrays it was built from, which
needs a synthetic fixture.  This is the other half of the same gate: a
trajectory a real run produced, where there is no reference to compare to, so
the question becomes whether independent readers agree about what is in it.

That is worth asking separately because the synthetic fixture is 37 atoms and
one frame per axis.  A real run is thousands of atoms over tens of frames, which
is where a header field that happens to be right for a small file -- a record
marker, an atom count that fits in one block, a box convention -- stops being
right.

Readers are whichever of MDAnalysis, MDTraj and pytraj import; no single
environment on this machine has all three, so run it once per interpreter.
pytraj is the one that also covers VMD, since cpptraj reads DCD through VMD's
own dcdplugin.

    python tools/crossread_dcd.py --dcd fixtures/run-artefacts/equil.dcd \\
        --pdb fixtures/run-artefacts/equil.pdb
"""

import argparse
import itertools
import pathlib

import numpy as np


def with_mdanalysis(dcd: pathlib.Path, pdb: pathlib.Path):
    import MDAnalysis

    universe = MDAnalysis.Universe(str(pdb), str(dcd))
    positions, boxes = [], []
    for _ in universe.trajectory:
        # MDAnalysis reports ångström; the engine's unit is nm.
        positions.append(universe.atoms.positions.copy() / 10.0)
        boxes.append(np.asarray(universe.dimensions[:3], dtype=float) / 10.0)
    return np.asarray(positions), np.asarray(boxes)


def with_mdtraj(dcd: pathlib.Path, pdb: pathlib.Path):
    import mdtraj

    trajectory = mdtraj.load_dcd(str(dcd), top=str(pdb))
    # MDTraj is already in nm.
    return np.asarray(trajectory.xyz, dtype=float), np.asarray(
        trajectory.unitcell_lengths, dtype=float
    )


def with_pytraj(dcd: pathlib.Path, pdb: pathlib.Path):
    import pytraj

    trajectory = pytraj.iterload(str(dcd), str(pdb))
    positions = np.asarray(trajectory.xyz, dtype=float) / 10.0
    boxes = np.asarray([frame.box.values[:3] for frame in trajectory]) / 10.0
    return positions, boxes


READERS = {
    "MDAnalysis": with_mdanalysis,
    "MDTraj": with_mdtraj,
    "pytraj": with_pytraj,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcd", required=True, type=pathlib.Path)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2e-6,
        # The file holds f32 coordinates, so two readers can differ only by how
        # they widen them and by the angstrom conversion. A nanometre coordinate
        # of order 3 has an f32 ulp near 2.4e-7; 2e-6 leaves room for the
        # conversion without admitting a real disagreement.
        help="nm; the most two readers may differ by (default 2e-6)",
    )
    args = parser.parse_args()

    results = {}
    for name, reader in READERS.items():
        try:
            results[name] = reader(args.dcd, args.pdb)
        except ImportError:
            print(f"  {name:<12} not installed here")
        except Exception as error:  # a reader that refuses the file is the finding
            print(f"  {name:<12} FAILED: {type(error).__name__}: {error}")

    if not results:
        raise SystemExit("no reader available; see tools/README.md")

    for name, (positions, boxes) in results.items():
        print(
            f"  {name:<12} {positions.shape[0]} frames, {positions.shape[1]} atoms, "
            f"box {np.round(boxes[0], 4).tolist()} nm"
        )

    worst = 0.0
    for (first, second) in itertools.combinations(sorted(results), 2):
        a, _ = results[first]
        b, _ = results[second]
        if a.shape != b.shape:
            raise SystemExit(f"  {first} and {second} disagree on shape: {a.shape} vs {b.shape}")
        difference = float(np.max(np.abs(a - b)))
        worst = max(worst, difference)
        print(f"  {first} vs {second}: max |dx| {difference:.3e} nm")

    if len(results) > 1:
        print(f"  worst pairwise difference {worst:.3e} nm (tolerance {args.tolerance:.1e})")
        if worst > args.tolerance:
            raise SystemExit("readers disagree by more than f32 round-off")
    print("  OK")


if __name__ == "__main__":
    main()
