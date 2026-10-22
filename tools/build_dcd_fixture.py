"""A reference DCD written by OpenMM, and the data that went into it.

DCD is a Fortran unformatted format with record markers, several incompatible
dialects, and a header patched as frames arrive.  Every way of getting it wrong
produces a file that *opens* -- nanometres instead of angstrom reads as a system
ten times too small, a raw timestep makes every time axis wrong by a constant
factor -- so the test for a writer has to be byte equality against a writer
known to be right, not "a reader accepted it".

This writes one with deterministic coordinates so the comparison is exact.  The
two 80-character title lines are the only part that cannot match: OpenMM puts
the current time in the second one.

    python tools/build_dcd_fixture.py --out fixtures/dcd
    python tools/build_dcd_fixture.py --shape dodecahedron \
        --out fixtures/dcd-dodecahedron

🔴 The second one is the only test that can catch the unit cell being written
as a *rectangle*.  A DCD does not store box vectors; it stores three edge
lengths and three angle cosines, in the order `[a, cos gamma, b, cos beta,
cos alpha, c]`.  For a rhombic dodecahedron the three edges are all equal and
two of the angles are 60 degrees, while the vectors' *diagonal* is
`(w, w, 0.707w)` -- so a writer handed the diagonal and assuming right angles
produces a cell with the right volume, the wrong shape, and no error anywhere.
Only byte equality against OpenMM shows it.
"""

import argparse
import json
import pathlib

import numpy as np
import openmm
from openmm import app, unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--atoms", type=int, default=37)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--interval", type=int, default=250)
    parser.add_argument("--first-step", type=int, default=1000)
    parser.add_argument("--shape", default="cube",
                        choices=["cube", "dodecahedron", "octahedron"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # A topology with the right atom count and a box, which is what DCDFile
    # reads out of it. The elements do not reach the file.
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("XXX", chain)
    for i in range(args.atoms):
        topology.addAtom(f"A{i}", app.element.carbon, residue)

    def vectors_for(width: float):
        """`Modeller._computeBoxVectors`, which is where these shapes come from."""
        if args.shape == "cube":
            return (openmm.Vec3(width, 0, 0), openmm.Vec3(0, width, 0),
                    openmm.Vec3(0, 0, width))
        if args.shape == "dodecahedron":
            return (openmm.Vec3(width, 0, 0), openmm.Vec3(0, width, 0),
                    openmm.Vec3(0.5, 0.5, 0.5 * np.sqrt(2)) * width)
        return (openmm.Vec3(width, 0, 0),
                openmm.Vec3(1 / 3, 2 * np.sqrt(2) / 3, 0) * width,
                openmm.Vec3(-1 / 3, np.sqrt(2) / 3, np.sqrt(6) / 3) * width)

    rng = np.random.default_rng(0xD0D)
    frames = []
    boxes = []
    vectors = []
    for frame in range(args.frames):
        # Coordinates spread over a few nanometres, with a deliberate negative
        # and a deliberate value needing more than f32 to hold exactly, so a
        # writer that rounds differently shows up.
        positions = rng.uniform(-2.0, 6.0, size=(args.atoms, 3))
        positions[0] = [-1.2345678901, 0.0, 3.14159265358979]
        if args.shape == "cube":
            box = [3.0 + 0.01 * frame, 4.0 + 0.02 * frame, 5.0 + 0.03 * frame]
            cell = (
                openmm.Vec3(box[0], 0, 0),
                openmm.Vec3(0, box[1], 0),
                openmm.Vec3(0, 0, box[2]),
            )
        else:
            # One width, because these shapes are defined by a single width --
            # and it still changes frame to frame, so a writer that caches the
            # cell is caught.
            cell = vectors_for(4.0 + 0.05 * frame)
            box = [cell[0][0], cell[1][1], cell[2][2]]
        frames.append(positions)
        boxes.append(box)
        vectors.append([component for row in cell for component in row])

    path = args.out / "reference.dcd"
    with open(path, "wb") as handle:
        topology.setPeriodicBoxVectors(
            tuple(openmm.Vec3(*vectors[0][row * 3:row * 3 + 3]) for row in range(3))
            * unit.nanometer
        )
        writer = app.DCDFile(
            handle,
            topology,
            args.timestep * unit.picosecond,
            firstStep=args.first_step,
            interval=args.interval,
        )
        for positions, cell in zip(frames, vectors):
            writer.writeModel(
                positions * unit.nanometer,
                periodicBoxVectors=tuple(
                    openmm.Vec3(*cell[row * 3:row * 3 + 3]) for row in range(3)
                )
                * unit.nanometer,
            )

    report = {
        "openmm": openmm.version.version,
        "shape": args.shape,
        "atoms": args.atoms,
        "frames": args.frames,
        "timestep_ps": args.timestep,
        "interval": args.interval,
        "first_step": args.first_step,
        "bytes": path.stat().st_size,
        # The two title lines are the only part a byte comparison must skip:
        # OpenMM writes the current time into the second one.
        "title_offset": 100,
        "title_bytes": 160,
        "note": (
            "Coordinates and box lengths are nm here; the file holds angstrom. "
            "dt in the header is ps/0.04888821, the AKMA time unit."
        ),
    }
    # A topology file too: pytraj and parmed need one to open a DCD at all,
    # because the format carries no atom names or connectivity.
    with open(args.out / "topology.pdb", "w") as handle:
        app.PDBFile.writeFile(
            topology, frames[0] * unit.nanometer, handle, keepIds=True
        )

    np.save(args.out / "positions.npy", np.array(frames).ravel())
    np.save(args.out / "boxes.npy", np.array(boxes).ravel())
    # The full 3x3 per frame. `boxes.npy` is only the diagonal, which is what a
    # writer must *not* mistake for the edge lengths.
    np.save(args.out / "boxvectors.npy", np.array(vectors).ravel())
    (args.out / "facts.json").write_text(json.dumps(report, indent=1))
    print(f"  wrote {path} ({report['bytes']} bytes), {args.frames} frames of {args.atoms} atoms")


if __name__ == "__main__":
    main()
