"""Every parameter OpenMM assigned, term by term, for parity comparison.

Phase 3's gate is term-by-term parameter parity, and the only way to get that is
to ask OpenMM what it assigned rather than to compare an energy.  An energy is a
sum: a wrong torsion and a wrong charge can cancel to within a tolerance and a
single number will not say which term is wrong, or that any is.

So this dumps the built `System` as sorted lists of (atoms, parameters) and the
topology it was built from.  A reimplementation compares row by row, and a
mismatch names the bond.

Sorted canonically -- atom tuples ordered the way the term's symmetry allows --
because the *order* terms are created in is an implementation detail and
comparing it would fail on a correct implementation.

    python tools/dump_parameters.py --pdb fixtures/prep/1L2Y-h.pdb \\
        --forcefield amber14/protein.ff14SB.xml --out fixtures/prep/1L2Y-h
"""

import argparse
import json
import pathlib

import numpy as np
import openmm
from openmm import app, unit


def canonical_bond(a, b):
    return (a, b) if a <= b else (b, a)


def canonical_angle(a, b, c):
    # The centre is fixed; the ends are interchangeable.
    return (a, b, c) if a <= c else (c, b, a)


def canonical_torsion(a, b, c, d):
    # A proper torsion read backwards is the same torsion.
    return (a, b, c, d) if (b, c) <= (c, b) and a <= d or b < c else (d, c, b, a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--forcefield", nargs="+", required=True)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    pdb = app.PDBFile(str(args.pdb))
    field = app.ForceField(*args.forcefield)
    # No cutoff, no constraints, no water rigidity: this is about parameter
    # assignment, and every one of those options *changes* what is in the
    # System. Comparing them would conflate assignment with system options.
    system = field.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
        removeCMMotion=False,
    )

    dump = {
        "pdb": args.pdb.name,
        "forcefield": args.forcefield,
        "openmm": openmm.version.version,
        "atoms": system.getNumParticles(),
        "masses": [
            system.getParticleMass(i).value_in_unit(unit.dalton)
            for i in range(system.getNumParticles())
        ],
        "topology": {
            "atoms": [
                [a.name, a.residue.name, a.residue.id, a.residue.chain.id,
                 None if a.element is None else a.element.symbol]
                for a in pdb.topology.atoms()
            ],
            "bonds": sorted(
                canonical_bond(b[0].index, b[1].index) for b in pdb.topology.bonds()
            ),
        },
    }

    for force in system.getForces():
        kind = type(force).__name__
        if kind == "HarmonicBondForce":
            rows = []
            for i in range(force.getNumBonds()):
                p1, p2, length, k = force.getBondParameters(i)
                rows.append(
                    list(canonical_bond(p1, p2))
                    + [length.value_in_unit(unit.nanometer),
                       k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2)]
                )
            dump["bonds"] = sorted(rows)
        elif kind == "HarmonicAngleForce":
            rows = []
            for i in range(force.getNumAngles()):
                p1, p2, p3, angle, k = force.getAngleParameters(i)
                rows.append(
                    list(canonical_angle(p1, p2, p3))
                    + [angle.value_in_unit(unit.radian),
                       k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2)]
                )
            dump["angles"] = sorted(rows)
        elif kind == "PeriodicTorsionForce":
            rows = []
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, n, phase, k = force.getTorsionParameters(i)
                rows.append(
                    [p1, p2, p3, p4, int(n),
                     phase.value_in_unit(unit.radian),
                     k.value_in_unit(unit.kilojoule_per_mole)]
                )
            # Not canonicalised by atom order: a torsion's direction is not free
            # once impropers are in the same force. Sorted as written instead,
            # which a comparison has to account for.
            dump["torsions"] = sorted(rows)
        elif kind == "NonbondedForce":
            dump["nonbonded"] = {
                "coulomb14scale": None,
                "particles": [
                    [
                        force.getParticleParameters(i)[0].value_in_unit(
                            unit.elementary_charge
                        ),
                        force.getParticleParameters(i)[1].value_in_unit(unit.nanometer),
                        force.getParticleParameters(i)[2].value_in_unit(
                            unit.kilojoule_per_mole
                        ),
                    ]
                    for i in range(force.getNumParticles())
                ],
                "exceptions": sorted(
                    list(canonical_bond(*force.getExceptionParameters(i)[:2]))
                    + [
                        force.getExceptionParameters(i)[2].value_in_unit(
                            unit.elementary_charge**2
                        ),
                        force.getExceptionParameters(i)[3].value_in_unit(unit.nanometer),
                        force.getExceptionParameters(i)[4].value_in_unit(
                            unit.kilojoule_per_mole
                        ),
                    ]
                    for i in range(force.getNumExceptions())
                ),
            }
        else:
            dump.setdefault("other_forces", []).append(kind)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.with_suffix(".json")).write_text(json.dumps(dump, indent=1))
    counts = {k: len(v) for k, v in dump.items() if isinstance(v, list)}
    print(f"  {dump['atoms']} atoms, {len(dump['topology']['bonds'])} topology bonds")
    print(f"  bonds {len(dump.get('bonds', []))}  angles {len(dump.get('angles', []))}  "
          f"torsions {len(dump.get('torsions', []))}  "
          f"exceptions {len(dump.get('nonbonded', {}).get('exceptions', []))}")
    print(f"  wrote {args.out.with_suffix('.json')}")
    if "other_forces" in dump:
        print(f"  other forces present: {dump['other_forces']}")


if __name__ == "__main__":
    main()
