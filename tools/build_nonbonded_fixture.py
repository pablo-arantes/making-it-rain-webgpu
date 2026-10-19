#!/usr/bin/env python
"""A small, hand-built nonbonded fixture, and the decomposition it pins down.

The solvated fixtures from `build_reference_system.py` are the acceptance gate,
but they are 0.7-7.5 MB, are not reproducible (docs/NOTEBOOK_PARITY.md finding
2) and are therefore not committed.  This builds a system small enough to
commit and fully specified here, so the nonbonded kernels have a reference that
CI can use with no conda environment.

It also answers the question the kernels cannot be written without: **which
force group does each Ewald term land in?**  OpenMM reports a direct-space
group and a reciprocal-space group, and the split is not the obvious one -- for
the 3HTB fixture direct space is +1.56e6 kJ/mol, which is far too positive to
be a pair sum over a water box.  This script decomposes the energy
independently, in numpy, and prints both, so the answer is measured rather than
assumed.

    python tools/build_nonbonded_fixture.py --out test/fixtures
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import openmm
from openmm import XmlSerializer, unit

KJ = unit.kilojoule_per_mole
ONE_4PI_EPS0 = 138.93545764498457   # kJ/mol nm / e^2

# A deliberately varied little system: both signs of charge, a particle with no
# LJ at all, a particle with no charge, and a range of sigmas.
PARTICLES = [
    # mass,  charge,  sigma,   epsilon
    (15.999, -0.834, 0.31507, 0.63639),   # water O
    (1.008, 0.417, 0.00000, 0.00000),     # water H, no LJ at all
    (1.008, 0.417, 0.00000, 0.00000),
    (15.999, -0.834, 0.31507, 0.63639),
    (1.008, 0.417, 0.00000, 0.00000),
    (1.008, 0.417, 0.00000, 0.00000),
    (22.990, 1.000, 0.24300, 0.36585),    # Na+
    (35.453, -1.000, 0.44777, 0.14891),   # Cl-
    (12.011, 0.000, 0.33997, 0.45773),    # uncharged carbon
    (14.007, -0.416, 0.32500, 0.71128),   # nitrogen
]
# Water internal pairs are excluded; the last two get a scaled 1-4 style
# exception, which is what an AMBER protein or an OpenFF ligand produces.
EXCEPTIONS = [
    (0, 1, 0.0, 1.0, 0.0),
    (0, 2, 0.0, 1.0, 0.0),
    (1, 2, 0.0, 1.0, 0.0),
    (3, 4, 0.0, 1.0, 0.0),
    (3, 5, 0.0, 1.0, 0.0),
    (4, 5, 0.0, 1.0, 0.0),
    # chargeProd, sigma, epsilon for a 1-4 pair: AMBER scales by 1/1.2 and 1/2
    (8, 9, 0.000 * -0.416 / 1.2, 0.5 * (0.33997 + 0.32500), math.sqrt(0.45773 * 0.71128) / 2),
]
BOX = 2.4            # nm, cubic
CUTOFF = 0.9         # nm, below half the box
TOLERANCE = 0.0005


def positions() -> np.ndarray:
    """Fixed coordinates, chosen to put pairs on both sides of the cutoff."""
    return np.array([
        [0.400, 0.400, 0.400],
        [0.496, 0.400, 0.400],
        [0.368, 0.491, 0.400],
        [1.100, 0.500, 0.450],
        [1.196, 0.500, 0.450],
        [1.068, 0.591, 0.450],
        [0.700, 1.300, 0.600],
        [1.600, 1.700, 1.900],   # beyond the cutoff from most of the rest
        [0.900, 0.900, 0.900],
        [1.040, 1.000, 0.980],
    ])


def build(method: int) -> tuple[openmm.System, openmm.NonbondedForce]:
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(BOX, 0, 0), openmm.Vec3(0, BOX, 0), openmm.Vec3(0, 0, BOX))
    nb = openmm.NonbondedForce()
    for mass, charge, sigma, epsilon in PARTICLES:
        system.addParticle(mass)
        nb.addParticle(charge, sigma, epsilon)
    for p1, p2, q, sigma, epsilon in EXCEPTIONS:
        nb.addException(p1, p2, q, sigma, epsilon)
    nb.setNonbondedMethod(method)
    nb.setCutoffDistance(CUTOFF)
    nb.setEwaldErrorTolerance(TOLERANCE)
    nb.setUseDispersionCorrection(True)
    system.addForce(nb)
    nb.setForceGroup(0)
    nb.setReciprocalSpaceForceGroup(1)
    return system, nb


def minimum_image(d: np.ndarray) -> np.ndarray:
    return d - BOX * np.round(d / BOX)


def decompose(alpha: float) -> dict:
    """Our own decomposition of the Ewald energy, in numpy, from first principles."""
    pos = positions()
    charge = np.array([p[1] for p in PARTICLES])
    sigma = np.array([p[2] for p in PARTICLES])
    epsilon = np.array([p[3] for p in PARTICLES])
    n = len(PARTICLES)
    excluded = {(min(a, b), max(a, b)) for a, b, *_ in EXCEPTIONS}

    direct_coulomb = 0.0
    direct_lj = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in excluded:
                continue
            d = minimum_image(pos[j] - pos[i])
            r = float(np.linalg.norm(d))
            if r >= CUTOFF:
                continue
            direct_coulomb += ONE_4PI_EPS0 * charge[i] * charge[j] * math.erfc(alpha * r) / r
            s = 0.5 * (sigma[i] + sigma[j])
            e = math.sqrt(epsilon[i] * epsilon[j])
            if e != 0.0:
                sr6 = (s / r) ** 6
                direct_lj += 4 * e * (sr6 * sr6 - sr6)

    # Every charge screens itself in reciprocal space; that is removed here.
    self_energy = -ONE_4PI_EPS0 * alpha / math.sqrt(math.pi) * float(np.sum(charge ** 2))

    # Reciprocal space also includes every excluded pair, so each one's
    # reciprocal contribution is subtracted back out.
    exclusion_correction = 0.0
    for a, b, *_ in EXCEPTIONS:
        d = minimum_image(pos[b] - pos[a])
        r = float(np.linalg.norm(d))
        exclusion_correction -= ONE_4PI_EPS0 * charge[a] * charge[b] * math.erf(alpha * r) / r

    # The exceptions' own interactions, unscreened.
    exception_energy = 0.0
    for a, b, q, s, e in EXCEPTIONS:
        d = minimum_image(pos[b] - pos[a])
        r = float(np.linalg.norm(d))
        if q != 0.0:
            exception_energy += ONE_4PI_EPS0 * q / r
        if e != 0.0:
            sr6 = (s / r) ** 6
            exception_energy += 4 * e * (sr6 * sr6 - sr6)

    # Brute-force reciprocal sum, so it can be checked against OpenMM's Ewald.
    kmax = 12
    volume = BOX ** 3
    recip = 0.0
    for nx in range(-kmax, kmax + 1):
        for ny in range(-kmax, kmax + 1):
            for nz in range(-kmax, kmax + 1):
                if nx == ny == nz == 0:
                    continue
                k = 2 * math.pi * np.array([nx, ny, nz]) / BOX
                ksq = float(k @ k)
                if ksq > (2 * math.pi * kmax / BOX) ** 2:
                    continue
                phase = pos @ k
                structure = complex(float(np.sum(charge * np.cos(phase))),
                                    float(np.sum(charge * np.sin(phase))))
                recip += (abs(structure) ** 2) * math.exp(-ksq / (4 * alpha ** 2)) / ksq
    recip *= 2 * math.pi / volume * ONE_4PI_EPS0

    # OpenMM's analytic long-range dispersion correction.
    pairs = 0.0
    for i in range(n):
        for j in range(n):
            s = 0.5 * (sigma[i] + sigma[j])
            e = math.sqrt(epsilon[i] * epsilon[j])
            pairs += e * s ** 6
    dispersion = -(2.0 / 3.0) * math.pi * pairs / (volume * CUTOFF ** 3)

    return {
        "alpha_per_nm": alpha,
        "direct_coulomb": direct_coulomb,
        "direct_lj": direct_lj,
        "self_energy": self_energy,
        "exclusion_correction": exclusion_correction,
        "exception_energy": exception_energy,
        "reciprocal_brute_force": recip,
        "dispersion_correction_naive": dispersion,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="test/fixtures")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    reference = openmm.Platform.getPlatformByName("Reference")
    report: dict = {"box_nm": BOX, "cutoff_nm": CUTOFF, "ewald_tolerance": TOLERANCE,
                    "particles": PARTICLES, "exceptions": EXCEPTIONS,
                    "positions_nm": positions().tolist(),
                    "openmm": openmm.version.version}

    for label, method in (("Ewald", openmm.NonbondedForce.Ewald),
                          ("PME", openmm.NonbondedForce.PME)):
        system, nb = build(method)
        context = openmm.Context(system, openmm.VerletIntegrator(0.001), reference)
        context.setPositions(positions() * unit.nanometer)
        if method == openmm.NonbondedForce.PME:
            alpha, nx, ny, nz = nb.getPMEParametersInContext(context)
            alpha_value = alpha.value_in_unit(alpha.unit) if hasattr(alpha, "unit") else alpha
        else:
            # Ewald has no grid, and OpenMM derives the same alpha from the
            # tolerance: alpha = sqrt(-log(2*tol))/cutoff.
            alpha_value = math.sqrt(-math.log(2 * TOLERANCE)) / CUTOFF
            nx = ny = nz = 0
        direct = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(KJ)
        recip = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(KJ)
        state = context.getState(getEnergy=True, getForces=True, getPositions=True)
        entry = {
            "alpha_per_nm": alpha_value,
            "grid": [nx, ny, nz],
            "direct_group": direct,
            "reciprocal_group": recip,
            "total": state.getPotentialEnergy().value_in_unit(KJ),
            "forces": state.getForces(asNumpy=True)
            .value_in_unit(KJ / unit.nanometer).tolist(),
        }
        report[label] = entry
        if label == "Ewald":
            # Only the exact method can be checked against a brute-force sum.
            ours = decompose(alpha_value)
            report["decomposition"] = ours
            print(f"\n--- {label}: alpha = {alpha_value:.9f} /nm ---")
            print(f"  OpenMM direct group      {direct:16.6f}")
            print(f"  OpenMM reciprocal group  {recip:16.6f}")
            print(f"  OpenMM total             {entry['total']:16.6f}")
            print("\n  our terms:")
            for key in ("direct_coulomb", "direct_lj", "exclusion_correction",
                        "exception_energy", "self_energy", "reciprocal_brute_force",
                        "dispersion_correction_naive"):
                print(f"    {key:28s} {ours[key]:16.6f}")
            # The hypothesis the kernels are written against.
            direct_hypothesis = (ours["direct_coulomb"] + ours["direct_lj"]
                                 + ours["exclusion_correction"] + ours["exception_energy"]
                                 + ours["dispersion_correction_naive"])
            recip_hypothesis = ours["reciprocal_brute_force"] + ours["self_energy"]
            print("\n  hypothesis: direct group = coulomb + lj + exclusion correction"
                  " + exceptions + dispersion")
            print(f"    ours {direct_hypothesis:16.6f}   openmm {direct:16.6f}"
                  f"   diff {direct_hypothesis - direct:.6e}")
            print("  hypothesis: reciprocal group = reciprocal sum + self energy")
            print(f"    ours {recip_hypothesis:16.6f}   openmm {recip:16.6f}"
                  f"   diff {recip_hypothesis - recip:.6e}")
        else:
            print(f"\n--- {label}: alpha = {alpha_value:.9f} /nm, grid {nx}x{ny}x{nz} ---")
            print(f"  OpenMM direct group      {direct:16.6f}")
            print(f"  OpenMM reciprocal group  {recip:16.6f}")
            print(f"  PME minus Ewald, reciprocal: "
                  f"{recip - report['Ewald']['reciprocal_group']:.6e}")
        # The committed fixture uses PME, which is what the notebook runs.
        if label == "PME":
            open(os.path.join(args.out, "nonbonded-system.xml"), "w").write(
                XmlSerializer.serialize(system))
            open(os.path.join(args.out, "nonbonded-state.xml"), "w").write(
                XmlSerializer.serialize(state))
        del context

    with open(os.path.join(args.out, "nonbonded-facts.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {args.out}/nonbonded-{{system,state}}.xml and nonbonded-facts.json")


if __name__ == "__main__":
    main()
