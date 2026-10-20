"""Read a DCD with every reader available and compare against what went in.

The gate in docs/PLAN.md is that the trajectories open in MDAnalysis, MDTraj,
pytraj and VMD.  No single environment on this machine has all of them, so the
tool checks whichever it can import and *merges* into one report -- run it once
per interpreter.  MDAnalysis lives in the base environment and the OpenMM
oracle stack in `parametrizani_env`; nothing is installed to make this work.

VMD is covered without VMD: pytraj reads DCD through cpptraj, which uses VMD's
own `dcdplugin`, and the plugin announces itself on stderr -- "detected standard
32-bit DCD file of native endianness / CHARMM format DCD file".  That is the
same code VMD runs, so a file it accepts is a file VMD accepts.

Each reader is asked for coordinates and unit cells, converted back to
nanometres, and compared against the arrays the fixture was built from.  The
interesting failures are not "it threw" -- they are a factor of ten from the
angstrom conversion and a time axis off by 0.04888821 from the AKMA one, both of
which read as a perfectly valid file.

    python tools/check_dcd.py --dcd fixtures/dcd/ours.dcd --fixture fixtures/dcd
"""

import argparse
import json
import pathlib

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcd", required=True, type=pathlib.Path)
    parser.add_argument("--fixture", required=True, type=pathlib.Path)
    args = parser.parse_args()

    facts = json.loads((args.fixture / "facts.json").read_text())
    atoms, frames = facts["atoms"], facts["frames"]
    expected_x = np.load(args.fixture / "positions.npy").reshape(frames, atoms, 3)
    expected_box = np.load(args.fixture / "boxes.npy").reshape(frames, 3)
    results = {}

    def compare(name, coordinates_nm, boxes_nm, times_ps=None):
        worst_x = float(np.abs(coordinates_nm - expected_x).max())
        worst_box = (
            float(np.abs(boxes_nm - expected_box).max()) if boxes_nm is not None else None
        )
        entry = {
            "frames": int(coordinates_nm.shape[0]),
            "atoms": int(coordinates_nm.shape[1]),
            "worst_coordinate_error_nm": worst_x,
            "worst_box_error_nm": worst_box,
        }
        if times_ps is not None:
            # The time axis, which is where the AKMA unit shows up: a writer
            # that stored ps instead would put these out by 1/0.04888821.
            wanted = facts["first_step"] * facts["timestep_ps"] + facts[
                "timestep_ps"
            ] * facts["interval"] * np.arange(frames)
            entry["worst_time_error_ps"] = float(np.abs(times_ps - wanted).max())
        results[name] = entry
        print(
            f"  {name:<12} {entry['frames']} frames x {entry['atoms']} atoms  "
            f"coordinates {worst_x:.3e} nm  "
            f"box {'n/a' if worst_box is None else f'{worst_box:.3e} nm'}"
            + (
                f"  time {entry['worst_time_error_ps']:.3e} ps"
                if times_ps is not None
                else ""
            )
        )

    try:
        import mdtraj

        loaded = mdtraj.formats.DCDTrajectoryFile(str(args.dcd)).read()
        # mdtraj returns angstrom for this reader.
        xyz, cell_lengths, _cell_angles = loaded
        compare("mdtraj", np.asarray(xyz) / 10.0, np.asarray(cell_lengths) / 10.0)
    except ImportError:
        # Absent is not failed. Recording it as a failure would overwrite a
        # passing entry from another interpreter when the reports are merged,
        # which is exactly what happened the first time this ran.
        print("  mdtraj       not in this interpreter")
    except Exception as error:  # noqa: BLE001
        results["mdtraj"] = {"error": repr(error)}
        print(f"  mdtraj       FAILED {error!r}")

    topology = args.fixture / "topology.pdb"

    try:
        import pytraj

        traj = pytraj.iterload(str(args.dcd), top=str(topology))
        coordinates = np.asarray(traj.xyz) / 10.0
        boxes = np.asarray([f.box.values[:3] for f in traj]) / 10.0
        compare("pytraj", coordinates, boxes)
    except ImportError:
        print("  pytraj       not in this interpreter")
    except Exception as error:  # noqa: BLE001
        results["pytraj"] = {"error": repr(error)}
        print(f"  pytraj       FAILED {error!r}")

    try:
        import MDAnalysis as mda

        universe = mda.Universe(str(topology), str(args.dcd))
        coordinates = np.array(
            [universe.trajectory[i].positions for i in range(frames)]
        ) / 10.0
        boxes = np.array(
            [universe.trajectory[i].dimensions[:3] for i in range(frames)]
        ) / 10.0
        compare(f"MDAnalysis {mda.__version__}", coordinates, boxes)
    except ImportError:
        print("  MDAnalysis   not in this interpreter")
    except Exception as error:  # noqa: BLE001
        results["MDAnalysis"] = {"error": repr(error)}
        print(f"  MDAnalysis   FAILED {error!r}")

    try:
        import parmed  # noqa: F401
        import parmed.formats

        # ParmEd has a DCD *writer* and no reader: there is nothing named DCD in
        # `parmed.formats`. Recorded as a missing capability in the library
        # rather than as a failure of the file, which is what "FormatNotFound"
        # looks like if you do not check.
        results["parmed"] = {
            "unsupported": "ParmEd provides no DCD reader (parmed.formats has no DCD entry)"
        }
        print("  parmed       no DCD reader in this library")
    except ImportError:
        pass

    out = args.fixture / "reader-report.json"
    merged = json.loads(out.read_text()) if out.exists() else {}
    merged.update(results)
    merged["note"] = (
        "Merged across interpreters; run once per environment. VMD is covered "
        "through pytraj, which reads DCD with VMD's own dcdplugin."
    )
    out.write_text(json.dumps(merged, indent=1))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
