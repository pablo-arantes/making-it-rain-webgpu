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
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # A topology with the right atom count and a box, which is what DCDFile
    # reads out of it. The elements do not reach the file.
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("XXX", chain)
    for i in range(args.atoms):
        topology.addAtom(f"A{i}", app.element.carbon, residue)

    rng = np.random.default_rng(0xD0D)
    frames = []
    boxes = []
    for frame in range(args.frames):
        # Coordinates spread over a few nanometres, with a deliberate negative
        # and a deliberate value needing more than f32 to hold exactly, so a
        # writer that rounds differently shows up.
        positions = rng.uniform(-2.0, 6.0, size=(args.atoms, 3))
        positions[0] = [-1.2345678901, 0.0, 3.14159265358979]
        box = [3.0 + 0.01 * frame, 4.0 + 0.02 * frame, 5.0 + 0.03 * frame]
        frames.append(positions)
        boxes.append(box)

    path = args.out / "reference.dcd"
    with open(path, "wb") as handle:
        topology.setPeriodicBoxVectors(
            (
                openmm.Vec3(boxes[0][0], 0, 0),
                openmm.Vec3(0, boxes[0][1], 0),
                openmm.Vec3(0, 0, boxes[0][2]),
            )
            * unit.nanometer
        )
        writer = app.DCDFile(
            handle,
            topology,
            args.timestep * unit.picosecond,
            firstStep=args.first_step,
            interval=args.interval,
        )
        for positions, box in zip(frames, boxes):
            writer.writeModel(
                positions * unit.nanometer,
                periodicBoxVectors=(
                    openmm.Vec3(box[0], 0, 0),
                    openmm.Vec3(0, box[1], 0),
                    openmm.Vec3(0, 0, box[2]),
                )
                * unit.nanometer,
            )

    report = {
        "openmm": openmm.version.version,
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
    (args.out / "facts.json").write_text(json.dumps(report, indent=1))
    print(f"  wrote {path} ({report['bytes']} bytes), {args.frames} frames of {args.atoms} atoms")


if __name__ == "__main__":
    main()
