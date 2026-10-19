"""OpenMM's own molecule decomposition, for the barostat to be checked against.

`MonteCarloBarostat` scales each molecule's centroid and translates its atoms
rigidly, so which atoms share a molecule decides the move.  Getting that
grouping wrong is not a rounding error: merging two molecules moves them as one
rigid body, and splitting one stretches it.

`context.getMolecules()` is the authority, so it is recorded here rather than
reasoned about.  Labels are canonicalised by each molecule's lowest atom index,
which makes the numbering a property of the system instead of an artefact of the
order the union-find happened to run in -- otherwise two correct decompositions
would compare unequal.

    python tools/capture_molecules.py --fixture fixtures/1L2Y
"""

import argparse
import collections
import json
import pathlib

import numpy as np
import openmm
from openmm import unit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True, type=pathlib.Path)
    args = parser.parse_args()

    system = openmm.XmlSerializer.deserialize(
        (args.fixture / "system.xml").read_text()
    )
    state = openmm.XmlSerializer.deserialize((args.fixture / "state.xml").read_text())
    integrator = openmm.VerletIntegrator(0.002 * unit.picosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setState(state)

    molecules = context.getMolecules()
    particles = system.getNumParticles()
    label = np.full(particles, -1, dtype=np.int32)
    # Sort by lowest atom index so the numbering is canonical.
    for index, atoms in enumerate(sorted(molecules, key=min)):
        for atom in atoms:
            label[atom] = index
    if (label < 0).any():
        raise SystemExit("getMolecules() did not cover every particle")

    sizes = collections.Counter(len(m) for m in molecules)
    box = state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    volume = box[0][0] * box[1][1] * box[2][2]

    report = {
        "fixture": args.fixture.name,
        "openmm": openmm.version.version,
        "particles": particles,
        "molecules": len(molecules),
        "largest_molecule": max(len(m) for m in molecules),
        "size_histogram": {str(k): v for k, v in sorted(sizes.items())},
        "box_nm": [box[i][i] for i in range(3)],
        "volume_nm3": volume,
        # What OpenMM would start a barostat at, for the reference to match.
        "initial_volume_scale": 0.01 * volume,
    }
    np.save(args.fixture / "molecule-labels.npy", label.astype(np.float64))
    (args.fixture / "molecule-facts.json").write_text(json.dumps(report, indent=1))
    print(
        f"  {report['molecules']} molecules, largest {report['largest_molecule']} atoms, "
        f"sizes {report['size_histogram']}"
    )


if __name__ == "__main__":
    main()
