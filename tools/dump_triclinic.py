"""OpenMM's own triclinic minimum image, pinned for the reference's test.

`src/reference/triclinic.ts` has to agree with OpenMM, not merely with itself,
and the interesting part is that the obvious implementation does not: the
sequential reduction that is exact for a cube overestimates distances by more
than a nanometre for a dodecahedron. So this dumps what OpenMM returns, for
every shape the notebook offers, and the test compares against that.

The distance is read out of a `CustomNonbondedForce("r")` under
`CutoffPeriodic`: its energy *is* the minimum-image separation OpenMM computed,
so nothing is inferred from a force field.

    python tools/dump_triclinic.py test/fixtures/triclinic-openmm.json
"""

from __future__ import annotations

import itertools
import json
import math
import sys

import numpy as np
import openmm
import openmm.unit as unit
from openmm.app.internal.unitcell import reducePeriodicBoxVectors

WIDTH = 3.0
CUTOFF = 1.0


def shape_vectors(shape: str, d: float) -> np.ndarray:
    if shape == "cube":
        return np.array([[d, 0, 0], [0, d, 0], [0, 0, d]])
    if shape == "dodecahedron":
        return np.array([[d, 0, 0], [0, d, 0], [d / 2, d / 2, d * math.sqrt(2) / 2]])
    if shape == "octahedron":
        return np.array(
            [
                [d, 0, 0],
                [d / 3, d * 2 * math.sqrt(2) / 3, 0],
                [-d / 3, d * math.sqrt(2) / 3, d * math.sqrt(6) / 3],
            ]
        )
    raise SystemExit(f"unknown shape {shape}")


def reduced(vectors: np.ndarray) -> np.ndarray:
    quantity = reducePeriodicBoxVectors(
        [openmm.Vec3(*row) for row in vectors] * unit.nanometer
    )
    return np.array([[v.x, v.y, v.z] for v in quantity.value_in_unit(unit.nanometer)])


def openmm_distances(box: np.ndarray, separations: np.ndarray) -> list[float | None]:
    system = openmm.System()
    system.addParticle(1.0)
    system.addParticle(1.0)
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in box])
    force = openmm.CustomNonbondedForce("r")
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(CUTOFF)
    force.addParticle([])
    force.addParticle([])
    system.addForce(force)
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.001),
        openmm.Platform.getPlatformByName("Reference"),
    )
    out: list[float | None] = []
    for d in separations:
        context.setPositions([openmm.Vec3(0, 0, 0), openmm.Vec3(*d)])
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        r = energy.value_in_unit(unit.kilojoule_per_mole)
        # Zero means the pair fell outside the cutoff, where there is nothing
        # to compare: OpenMM reports no interaction rather than a distance.
        out.append(None if r == 0.0 else r)
    return out


def brute(d: np.ndarray, box: np.ndarray, span: int = 3) -> float:
    a, b, c = box
    return min(
        float(np.linalg.norm(d + i * a + j * b + k * c))
        for i, j, k in itertools.product(range(-span, span + 1), repeat=3)
    )


def main() -> None:
    destination = sys.argv[1] if len(sys.argv) > 1 else None
    rng = np.random.default_rng(20261022)
    cases = []
    for shape in ("cube", "dodecahedron", "octahedron"):
        raw = shape_vectors(shape, WIDTH)
        box = reduced(raw)
        separations = rng.uniform(-2 * WIDTH, 2 * WIDTH, size=(500, 3))
        distances = openmm_distances(box, separations)
        inside = [
            {
                "separation": [float(v) for v in d],
                "openmm": r,
                "brute": brute(d, box),
            }
            for d, r in zip(separations, distances)
            if r is not None
        ]
        cases.append(
            {
                "shape": shape,
                "width_nm": WIDTH,
                "cutoff_nm": CUTOFF,
                "requested_vectors": raw.tolist(),
                "reduced_vectors": box.tolist(),
                "already_reduced": bool(np.allclose(box, raw)),
                "volume_nm3": float(abs(np.linalg.det(box))),
                "volume_vs_cube": float(abs(np.linalg.det(box)) / WIDTH**3),
                "pairs_inside_cutoff": len(inside),
                "pairs": inside,
            }
        )
    report = {
        "what": "OpenMM's triclinic minimum image, for src/reference/triclinic.ts "
        "to be tested against. `openmm` is the separation OpenMM computed, read "
        "out of a CustomNonbondedForce('r') under CutoffPeriodic; `brute` is the "
        "minimum over 7^3 lattice images, so the two cross-check each other.",
        "openmm_version": openmm.version.version,
        "cases": cases,
    }
    text = json.dumps(report, indent=1)
    if destination is None:
        print(text)
    else:
        with open(destination, "w") as handle:
            handle.write(text)
        for case in cases:
            worst = max(abs(p["openmm"] - p["brute"]) for p in case["pairs"])
            print(
                f"{case['shape']:14s} {case['pairs_inside_cutoff']:4d} pairs, "
                f"volume {case['volume_vs_cube']:.3f} of the cube, "
                f"OpenMM vs brute force {worst:.2e} nm"
            )


if __name__ == "__main__":
    main()
