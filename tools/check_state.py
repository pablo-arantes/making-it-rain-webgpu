"""Can OpenMM read a State this engine wrote?

The protocol writes a serialised `State` after every stage and stride, which is
the notebook's own checkpoint mechanism -- a stride whose `.rst` exists is
skipped, and that is how a disconnected Colab session resumes.  So the file has
to be readable by OpenMM, or a run started in the browser cannot be continued in
Python and a half-finished run cannot be picked up at all.

Deserialised and compared bit for bit: these are decimal representations of
doubles, and the only thing that matters is that they read back to the same
bits.  OpenMM's own writer uses a different shortest form -- `.006` rather than
`0.006` -- so comparing the text would be comparing float printers.

    python tools/check_state.py --state fixtures/state-roundtrip/ours.xml \\
        --expected fixtures/state-roundtrip/expected.json
"""

import argparse
import json
import pathlib

import numpy as np
import openmm
from openmm import unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--expected", required=True, type=pathlib.Path)
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text())
    state = openmm.XmlSerializer.deserialize(args.state.read_text())

    positions = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    ).ravel()
    velocities = np.asarray(
        state.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
    ).ravel()
    box = np.asarray(
        state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
    ).ravel()

    report = {
        "openmm": openmm.version.version,
        "step_count": int(state.getStepCount()),
        "time_ps": float(state.getTime().value_in_unit(unit.picosecond)),
        # Exact, not approximate: a decimal that round-trips a double reads back
        # to the same double.
        "positions_exact": bool(
            np.array_equal(positions, np.array(expected["positions"]))
        ),
        "velocities_exact": bool(
            np.array_equal(velocities, np.array(expected["velocities"]))
        ),
        "box_exact": bool(np.array_equal(box, np.array(expected["box"]))),
        "step_count_exact": int(state.getStepCount()) == expected["stepCount"],
        "time_exact": float(state.getTime().value_in_unit(unit.picosecond))
        == expected["time"],
    }
    report["ok"] = all(
        report[key]
        for key in (
            "positions_exact",
            "velocities_exact",
            "box_exact",
            "step_count_exact",
            "time_exact",
        )
    )
    print(json.dumps(report, indent=1))
    if not report["ok"]:
        worst_x = float(np.abs(positions - np.array(expected["positions"])).max())
        worst_v = float(np.abs(velocities - np.array(expected["velocities"])).max())
        print(f"  worst position difference {worst_x:.3e} nm")
        print(f"  worst velocity difference {worst_v:.3e} nm/ps")
        raise SystemExit(1)
    print("  OpenMM read it back exactly")


if __name__ == "__main__":
    main()
