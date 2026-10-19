#!/usr/bin/env python
"""Build the reference (oracle) system for making-it-rain-webgpu.

This runs the *topology* half of reference/Uni_Dock+MD.ipynb -- cell 17 plus the
receptor preparation of cell 7 -- with the docking removed.  The ligand comes
from a HETATM residue of the PDB entry, with bond orders assigned from the RCSB
Chemical Component Dictionary, which is exactly the route the browser takes.

Outputs, per case, into <outdir>/<case>/:
    receptor.pdb    chain selection + OXT/H stripped + PDBFixer
    ligand.sdf      CCD-templated ligand with explicit hydrogens
    system.pdb      solvated complex (the notebook's system.pdb)
    system.xml      serialised openmm.System      <- the engine contract
    state.xml       serialised openmm.State       <- positions + box
    facts.json      atom/residue counts, force inventory, PME parameters,
                    per-force-group Reference-platform energies and forces

Environment: openmm >= 8.2, openmmforcefields, openff-toolkit, pdbfixer,
parmed, rdkit, and AmberTools (sqm/antechamber) on PATH for any gaff-* ligand
force field.  See tools/README.md.

    python tools/build_reference_system.py --pdb-id 3HTB --ligand JZ4 \
        --out fixtures/ --ligand-ff gaff-2.11
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

RCSB_PDB = "https://files.rcsb.org/download/{}.pdb"
RCSB_CCD_SDF = "https://files.rcsb.org/ligands/download/{}_ideal.sdf"

PROTEIN_FF = {
    # UI label -> (what the notebook's string selects, what it actually is)
    "ff14SB": "amber14/protein.ff14SB.xml",
    "ff99SB-ILDN": "amber99sbildn.xml",
    "ff99SB": "amber99sb.xml",
}
WATER_FF = {
    ("ff14SB", "tip3p"): "amber14/tip3p.xml",
    ("ff14SB", "spce"): "amber14/spce.xml",
    ("ff99SB-ILDN", "tip3p"): "tip3p.xml",
    ("ff99SB-ILDN", "spce"): "spce.xml",
    ("ff99SB", "tip3p"): "tip3p.xml",
    ("ff99SB", "spce"): "spce.xml",
}


def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def split_chains(pdb_path: str, chains: list[str], ligand_code: str | None,
                 receptor_out: str, ligand_out: str) -> tuple[int, int]:
    """Cell 7: keep selected chains, drop waters, drop OXT and existing H.

    Only the first MODEL is read, which is what PDBFixer and py3Dmol show for an
    NMR ensemble.  Alternate locations are passed through and resolved later by
    openmm.app.PDBFile, which keeps the first altLoc of each atom -- the same
    thing that happens in the notebook, where biopandas does not filter them.
    """
    prot, lig = [], []
    for line in open(pdb_path):
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM", "HETATM")):
            continue
        chain, resn, name = line[21], line[17:20].strip(), line[12:16].strip()
        if chains and chain not in chains:
            continue
        if resn == "HOH":
            continue
        if line.startswith("ATOM"):
            if name == "OXT" or line[76:78].strip() == "H":
                continue
            prot.append(line)
        elif ligand_code is not None and resn == ligand_code:
            lig.append(line)
    open(receptor_out, "w").write("".join(prot) + "END\n")
    if lig:
        open(ligand_out, "w").write("".join(lig) + "END\n")
    return len(prot), len(lig)


def run_pdbfixer(receptor: str, ph: float) -> dict:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=receptor)
    report = {"atoms_as_read_by_openmm": fixer.topology.getNumAtoms()}
    fixer.removeHeterogens()
    fixer.findMissingResidues()
    report["missingResidues"] = {str(k): v for k, v in fixer.missingResidues.items()}
    fixer.findMissingAtoms()
    report["missingAtoms"] = {str(k): [a.name for a in v] for k, v in fixer.missingAtoms.items()}
    report["missingTerminals"] = {str(k): v for k, v in fixer.missingTerminals.items()}
    fixer.addMissingAtoms()
    report["atoms_after_addMissingAtoms"] = fixer.topology.getNumAtoms()
    fixer.addMissingHydrogens(pH=ph)
    report["atoms_after_addMissingHydrogens"] = fixer.topology.getNumAtoms()
    PDBFile.writeFile(fixer.topology, fixer.positions, open(receptor, "w"))
    return report


def ligand_from_ccd(frag_pdb: str, code: str, workdir: str) -> tuple[str, dict]:
    """Assign bond orders from the CCD ideal coordinates, then add hydrogens."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    ideal = fetch(RCSB_CCD_SDF.format(code), os.path.join(workdir, f"{code}_ideal.sdf"))
    template = Chem.MolFromMolFile(ideal, removeHs=True, sanitize=True)
    if template is None:
        raise SystemExit(f"CCD component {code} did not parse; supply an SDF instead")
    frag = Chem.MolFromPDBFile(frag_pdb, removeHs=True, sanitize=False)
    report = {
        "ccd_code": code,
        "ccd_smiles": Chem.MolToSmiles(template),
        "ccd_heavy_atoms": template.GetNumAtoms(),
        "pdb_fragment_heavy_atoms": frag.GetNumAtoms(),
    }
    mol = AllChem.AssignBondOrdersFromTemplate(template, frag)
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    report["smiles_after_assignment"] = Chem.MolToSmiles(Chem.RemoveHs(mol))
    report["atoms_with_hydrogens"] = mol.GetNumAtoms()
    report["formal_charge"] = Chem.GetFormalCharge(mol)
    out = os.path.join(workdir, "ligand.sdf")
    Chem.MolToMolFile(mol, out)
    return out, report


def build(args) -> dict:
    import numpy as np
    import openmm
    import openmm.app as app
    from openmm import XmlSerializer, unit
    from openff.toolkit.topology import Molecule
    from openff.units.openmm import to_openmm
    from openmmforcefields.generators import GAFFTemplateGenerator, SMIRNOFFTemplateGenerator

    case = os.path.join(args.out, args.case or (args.pdb_id or "case"))
    os.makedirs(case, exist_ok=True)
    src = args.pdb_file or fetch(RCSB_PDB.format(args.pdb_id),
                                 os.path.join(case, f"{args.pdb_id}.pdb"))

    receptor = os.path.join(case, "receptor.pdb")
    frag = os.path.join(case, "ligand_fragment.pdb")
    n_prot, n_lig = split_chains(src, args.chains, args.ligand, receptor, frag)
    facts = {
        "source": os.path.basename(src),
        "chains": args.chains,
        "options": {
            "protein_ff": args.protein_ff, "water": args.water,
            "padding_angstrom": args.padding, "ions": args.ions,
            "concentration_molar": args.concentration, "pH": args.ph,
            "ligand_ff": args.ligand_ff,
        },
        "receptor_records_after_cell7_filter": n_prot,
        "ligand_records": n_lig,
        "pdbfixer": run_pdbfixer(receptor, args.ph),
    }

    ff_files = [PROTEIN_FF[args.protein_ff], WATER_FF[(args.protein_ff, args.water)]]
    forcefield = app.ForceField(*ff_files)
    facts["forcefield_files"] = ff_files

    molecules = []
    if n_lig:
        ligsdf, ligreport = ligand_from_ccd(frag, args.ligand, case)
        facts["ligand"] = ligreport
        ligand = Molecule.from_file(ligsdf)
        molecules = [ligand]
        gen_cls = GAFFTemplateGenerator if args.ligand_ff.startswith("gaff") \
            else SMIRNOFFTemplateGenerator
        t0 = time.time()
        forcefield.registerTemplateGenerator(
            gen_cls(forcefield=args.ligand_ff, molecules=molecules).generator)
        facts["ligand"]["template_generator"] = gen_cls.__name__
        facts["ligand"]["charge_seconds"] = round(time.time() - t0, 2)

    pdb = app.PDBFile(receptor)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    if molecules:
        modeller.add(molecules[0].to_topology().to_openmm(),
                     to_openmm(molecules[0].conformers[0]))
    facts["atoms_before_addHydrogens"] = modeller.topology.getNumAtoms()
    modeller.addHydrogens(forcefield, pH=args.ph)
    facts["atoms_after_addHydrogens"] = modeller.topology.getNumAtoms()
    modeller.addSolvent(forcefield, model=args.water,
                        padding=args.padding * unit.angstrom,
                        ionicStrength=args.concentration * unit.molar,
                        positiveIon="Na+" if args.ions == "NaCl" else "K+",
                        negativeIon="Cl-")
    topology = modeller.getTopology()
    # cell 17 shifts the whole system by +30 A before writing system.pdb
    positions = modeller.getPositions() + np.array([30, 30, 30]) * unit.angstrom

    counts: dict[str, int] = {}
    for r in topology.residues():
        counts[r.name] = counts.get(r.name, 0) + 1
    box = topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)
    facts.update({
        "atoms_total": topology.getNumAtoms(),
        "residues_total": topology.getNumResidues(),
        "residue_counts": counts,
        "n_waters": counts.get("HOH", 0),
        "n_positive_ions": counts.get("NA", 0) + counts.get("K", 0),
        "n_chloride": counts.get("CL", 0),
        "box_nm": [[round(v, 6) for v in row] for row in box],
    })
    app.PDBFile.writeFile(topology, positions, open(os.path.join(case, "system.pdb"), "w"))

    system = forcefield.createSystem(
        topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)

    inventory = []
    for i, f in enumerate(system.getForces()):
        entry = {"index": i, "class": f.__class__.__name__}
        for attr, key in (("getNumBonds", "numBonds"), ("getNumAngles", "numAngles"),
                          ("getNumTorsions", "numTorsions"),
                          ("getNumParticles", "numParticles"),
                          ("getNumExceptions", "numExceptions")):
            if hasattr(f, attr):
                entry[key] = getattr(f, attr)()
        if isinstance(f, openmm.NonbondedForce):
            entry.update({
                "cutoff_nm": f.getCutoffDistance().value_in_unit(unit.nanometer),
                "ewaldErrorTolerance": f.getEwaldErrorTolerance(),
                "useDispersionCorrection": f.getUseDispersionCorrection(),
                "useSwitchingFunction": f.getUseSwitchingFunction(),
            })
        inventory.append(entry)
    facts["forces"] = inventory
    facts["constraints"] = system.getNumConstraints()

    # one force group per force, with PME reciprocal space split out
    names = []
    for i, f in enumerate(system.getForces()):
        f.setForceGroup(i)
        names.append(f.__class__.__name__)
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    nb.setReciprocalSpaceForceGroup(len(names))
    names.append("NonbondedForce[reciprocal]")

    context = openmm.Context(system, openmm.VerletIntegrator(0.002),
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    alpha, nx, ny, nz = nb.getPMEParametersInContext(context)
    facts["pme"] = {
        "alpha_per_nm": alpha.value_in_unit(alpha.unit) if hasattr(alpha, "unit") else alpha,
        "grid_reference_platform": [nx, ny, nz],
    }
    energies = {}
    for g, name in enumerate(names):
        energies[f"{g}:{name}"] = context.getState(getEnergy=True, groups={g}) \
            .getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energies["total"] = context.getState(getEnergy=True).getPotentialEnergy() \
        .value_in_unit(unit.kilojoule_per_mole)
    facts["reference_energies_kJ_mol"] = energies

    state = context.getState(getPositions=True, getVelocities=True, getForces=True)
    forces_np = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    facts["reference_forces_kJ_mol_nm"] = {
        "rms": float((forces_np ** 2).sum(axis=1).mean() ** 0.5),
        "max_component": float(abs(forces_np).max()),
    }
    open(os.path.join(case, "system.xml"), "w").write(XmlSerializer.serialize(system))
    open(os.path.join(case, "state.xml"), "w").write(XmlSerializer.serialize(state))
    facts["system_xml_bytes"] = os.path.getsize(os.path.join(case, "system.xml"))

    import openmm.version
    facts["versions"] = {"openmm": openmm.version.version}
    for mod in ("openmmforcefields", "openff.toolkit", "rdkit", "parmed", "pdbfixer"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            facts["versions"][mod] = getattr(m, "__version__", "?")
        except Exception:
            facts["versions"][mod] = "missing"
    json.dump(facts, open(os.path.join(case, "facts.json"), "w"), indent=1, sort_keys=False)
    return facts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb-id")
    src.add_argument("--pdb-file")
    p.add_argument("--case", help="output subdirectory name (default: the PDB id)")
    p.add_argument("--chains", default="A",
                   help="comma-separated chain ids, empty string for all")
    p.add_argument("--ligand", default=None,
                   help="three-letter CCD code of the HETATM residue to keep")
    p.add_argument("--out", default="fixtures")
    p.add_argument("--protein-ff", default="ff14SB", choices=sorted(PROTEIN_FF))
    p.add_argument("--water", default="tip3p", choices=["tip3p", "spce"])
    p.add_argument("--padding", type=float, default=4.0, help="angstrom")
    p.add_argument("--ions", default="NaCl", choices=["NaCl", "KCl"])
    p.add_argument("--concentration", type=float, default=0.15, help="molar")
    p.add_argument("--ph", type=float, default=7.4)
    p.add_argument("--ligand-ff", default="gaff-2.11")
    args = p.parse_args()
    args.chains = [c for c in args.chains.split(",") if c]
    facts = build(args)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
