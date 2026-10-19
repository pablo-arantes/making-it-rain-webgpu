# Notebook parity

Every user-facing field of `reference/Uni_Dock+MD.ipynb`, what it does in
OpenMM/OpenFF/pytraj, and where it lands in the web app.

**Status column.** `P2`–`P5` name the phase in [PLAN.md](PLAN.md) that delivers
it (P2 = run an uploaded OpenMM system, P3 = protein in water from a PDB ID,
P4 = protein–ligand, P5 = polish). `dropped` means deliberately not ported,
with the reason given. A field whose behaviour differs from its label is marked
🔴 and explained in [Measured differences](#measured-differences-between-the-notebooks-labels-and-its-behaviour).

Everything in this file about the notebook's *behaviour* (as opposed to its
source) was measured on this machine with `tools/build_reference_system.py` and
`tools/probe_restraint_mask.py`; the records are in
[`measurements/`](measurements/).

---

## Out of scope by instruction: the docking half

Cells 8–16 and 31–33 are docking and are not ported. They are listed so the
boundary is explicit, not to be revisited later.

| Notebook cell | What it does | Why it is dropped |
| --- | --- | --- |
| 8 — P2Rank pockets | Java binary, surface ML pocket prediction | docking |
| 9 — pocket/residue selection | picks the reference residues for the box | docking |
| 11 — grid box (`Extending_value`) | PyMOL `get_extent` → Vina box | docking |
| 12 — ligand input (`Type`, `smiles_or_filename`) | SMILES/PDB/MOL → RDKit embed → MMFF | the web app's ligand comes from the structure the user provides, not from docking |
| 13 — TorchANI optimisation (`model_name`, `opt_tol`) | ANI-1x/1ccx/2x geometry optimisation, RHO reliability index | docking; also PyTorch |
| 14 — ligand 2D/3D depiction | RDKit drawing | replaced by the structure viewer |
| 15 — Uni-Dock (`seed`, `scoring`, `search_mode`, `exhaustiveness`, `max_step`, `num_modes`) | GPU docking | docking |
| 16 — best pose (`mode_number`) | pose selection from the docking output | docking |
| 31–32 — GNINA rescoring (`minimization`, `Skip`) | CNN affinity over the MD ensemble | needs the `gnina` CUDA binary; no browser route |
| 33 — best pose from MD (`Frame`) | writes one trajectory frame + prints GNINA scores | **the frame extraction is kept** (P4); the GNINA score print is dropped |
| 1–2, 4 — condacolab, pixi/pip installs, Google Drive mount | environment | no environment to install; files go to OPFS and the user's disk |
| 5 — `nvidia-smi` GPU check | confirms a Colab GPU was allocated | replaced by the WebGPU adapter panel (P2) |

---

## Structure input and cleaning (cell 7)

| Field | Default | Options | What it does | Web app |
| --- | --- | --- | --- | --- |
| `Google_Drive_Path` | `/content/` | string | working directory for every output | dropped — outputs go to OPFS, downloads, and optionally the File System Access API |
| `PDB_Source` | `PDB_ID` | `PDB_ID`, `Uploaded_PDB` | chooses download vs local file | P3 |
| `Query_PDB_ID` | `3HTB` | string | `GET https://files.rcsb.org/download/{id}.pdb` | P3 — the only structure request the app makes |
| `pdb_file` | `3D11.pdb` | string | reads an uploaded PDB | P3 — file picker / drag and drop |
| `selected_chains` | `['A']` | raw list | biopandas filter on `chain_id` for both ATOM and HETATM; empty list means all chains | P3 — multi-select built from the chains found in the file |
| *(implicit)* | | | drops `HETATM` with `residue_name == HOH` | P3 |
| *(implicit)* | | | drops atoms named `OXT` and atoms whose `element_symbol` is `H` | P3 — 🔴 see finding 2 |
| *(implicit)* | | | `PDBFixer.removeHeterogens()` | P3 |
| *(implicit)* | | | `findMissingResidues()` → `findMissingAtoms()` → `addMissingAtoms()` | P3 — loop/terminal rebuilding; on 3HTB chain A this re-adds exactly the `OXT` just deleted |
| *(implicit)* | | | `addMissingHydrogens(pH=7.4)` | P3 — 🔴 **pH is hardcoded here**, see finding 1 |
| *(implicit)* | | | `obabel -o pdbqt --partialcharge` | dropped — docking |
| *(implicit)* | | | prints the residue list (`name_residues.txt`) used later to pick a ligand and to index residues | P3/P4 — becomes the heterogen picker and the residue index used by the distance analyses |

## Topology (cell 17)

| Field | Default | Options | What it does | Web app |
| --- | --- | --- | --- | --- |
| `Force_field` | `AMBER14SB` | `AMBER14SB`, `AMBER99`, `AMBER99SB` | loads `amber14/protein.ff14SB.xml`, `amber99sb.xml`, `amber99sbildn.xml` respectively | P3 — 🔴 relabelled **ff14SB / ff99SB / ff99SB-ILDN**, see finding 3 |
| `Water_type` | `TIP3P` | `TIP3P`, `SPC/E` | with ff14SB: `amber14/tip3p.xml` / `amber14/spce.xml`; otherwise `tip3p.xml` / `spce.xml`. Both 3-site, so no virtual sites | P3 |
| `Padding_distance` | `4` | slider 1–10 Å | `Modeller.addSolvent(padding=…)`: box width = `max(2·radius + padding, 2·padding)` where `radius` is the bounding-**sphere** radius about the bounding-box centre — so padding is the **total** added width, padding/2 per side | P3 — 🔴 neither the tooltip nor the markdown describes this, see finding 4 |
| `Ions` | `NaCl` | `NaCl`, `KCl` | `positiveIon='Na+'` or `'K+'`; `negativeIon` is always `Cl-` | P3 |
| `Concentration` | `0.15` | string, molar | `ionicStrength`; OpenMM adds neutralising counter-ions first, then ion pairs for the requested strength | P3 |
| `pH` | `7.4` | string | `Modeller.addHydrogens(forcefield, pH=…)` | P3 — 🔴 no effect on this path, see finding 1 |
| `Ligand_Force_field` | `gaff-2.11` | 5 `gaff-*` + 11 `smirnoff99Frosst-*` + 13 `openff-*` | prefix selects `GAFFTemplateGenerator` or `SMIRNOFFTemplateGenerator`; both charge with AM1-BCC through AmberTools `sqm`, and the GAFF path additionally needs `antechamber` for **atom typing** | P4 — **the app's default becomes `openff-2.3.0`**, since `gaff-2.11` cannot run in a tab. The `openff-*` and `smirnoff99Frosst-*` families run in the browser with NAGL charges; `gaff-*` is kept as an option but upload-only. The notebook's list is also stale: `openff-2.1.x`/`2.2.x`/`2.3.0` and `gaff-2.2.20` now exist. See [PLAN.md](PLAN.md#ligand-force-field-and-charges) |
| *(implicit)* | | | `Modeller.add(ligand)` → `addHydrogens` → `addSolvent` | P3/P4 |
| *(implicit)* | | | `positions + [30, 30, 30] Å` | P3 — 🔴 cosmetic, see finding 5 |
| *(implicit)* | | | `PDBFile.writeFile` → `system.pdb` | P3 |
| *(implicit)* | | | second `createSystem(flexibleConstraints=True)` → ParmEd → `system_amber.prmtop` / `.inpcrd` | P3/P4 — needed anyway by the analysis path |
| *(implicit)* | | | same ParmEd structure → `system_gromacs.top` / `.gro` | P5, best effort — ParmEd's GROMACS writer has its own conventions; the files will be marked non-authoritative unless they round-trip |

## Viewer and interaction network (cells 19, 20)

| Field | Default | Options | What it does | Web app |
| --- | --- | --- | --- | --- |
| `color` | `gray` | `gray`, `rainbow` | py3Dmol cartoon colouring | P3 — 3Dmol.js |
| `show_sidechains` / `show_mainchains` | `False` | bool | adds stick styles | P3 |
| `show_ligand` | `True` | bool | sticks on `resn UNK`, greenCarbon | P4 |
| `show_box` | `True` | bool | 🔴 adds a **solvent-accessible surface**, not the periodic box | P3 — the app draws the real box and offers the surface separately |
| `box_opacity` | `0.6` | slider 0–1 | surface opacity | P3 |
| cell 20 — initial `LigNetwork` | — | — | ProLIF `Fingerprint().run(u.trajectory[::10], lig, prot)` on the single-frame `system.pdb`, `kind="aggregate"`, `rotation=270` | P4 — reimplemented in TS, with its own parity test against ProLIF |

## Equilibration (cells 22, 23) — NPT

| Field | Default | Options | What it does | Web app |
| --- | --- | --- | --- | --- |
| `Jobname` | `prot_lig_equil` | string | output basename | P2 |
| `Minimization_steps` | `10000` | 1000–100000 | `minimizeEnergy(tolerance=10 kJ/mol/nm, maxIterations=…)`, **with the restraints already added** | P2 |
| `Time` | `0.5` | string, ns | `nsteps = Time·1000 / dt` | P2 |
| `Integration_timestep` | `2` | `0.5`, `1`, `2`, `3`, `4` fs | integrator step | P2 — kept, with a warning above 2 fs and optional HMR (off by default) |
| `Temperature` | `298` | string, K | Langevin and barostat temperature | P2 |
| `Pressure` | `1` | string, bar | `MonteCarloBarostat(pressure, temperature)`, default frequency **25** steps | P2 |
| `Force_constant` | `700` | slider 0–2000, step 100 | `CustomExternalForce('k*periodicdistance(x,y,z,x0,y0,z0)^2')`, per-particle `k, x0, y0, z0`; `k=0` skips the force entirely | P2 — 🔴 labelled kJ/mol, the unit is **kJ/mol/nm²** |
| `Write_the_trajectory` | `10` | 10–1000 ps | `DCDReporter` interval | P2 |
| `Write_the_log` | `10` | 10–1000 ps | `StateDataReporter`: step, KE, PE, total E, T, volume, speed | P2 |
| *(implicit)* | | | `nonbondedMethod=PME`, `nonbondedCutoff=1.0 nm`, `ewaldErrorTolerance=0.0005`, `constraints=HBonds`, `rigidWater=True` | P1 — measured: α = 2.628261 nm⁻¹, grid 53³ for 3HTB and 25³ for 1L2Y on the Reference platform |
| *(implicit)* | | | `LangevinIntegrator(T, 1.0/ps, dt)`, `setConstraintTolerance(1e-6)` | P1 — 🔴 this is **LFMiddle** in OpenMM ≥ 8.2, see finding 6 |
| *(implicit)* | | | restraint atom list from pytraj `'!(:H*) & !(:HOH) & !(:NA) & !(:CL) & !(:K)'` | P2 — 🔴 **not heavy atoms**, see finding 7 |
| *(implicit)* | | | `setVelocitiesToTemperature(T)` after minimisation, unseeded | P2 — the app records the seed it used in `run.json` |
| *(implicit)* | | | writes `.rst` (serialised `State`), `.pdb` (last frame), `.dcd`, `.log` | P2 |

## Production (cells 25, 26) — NPT

| Field | Default | Options | What it does | Web app |
| --- | --- | --- | --- | --- |
| `Equilibrated_PDB` | `prot_lig_equil.pdb` | string | topology for the analysis stage | P2 |
| `State_file` | `prot_lig_equil.rst` | string | `context.setState(XmlSerializer.deserialize(…))` — positions, velocities and box | P2 |
| `Jobname` | `prot_lig_prod` | string | output basename | P2 |
| `Stride_Time` | `1` | string, ns | length of one stride | P2 |
| `Number_of_strides` | `1` | string | total time = strides × stride time; one `.dcd`/`.log`/`.rst`/`.pdb` per stride; a stride whose `.rst` exists is skipped, which is how a 12 h Colab session resumes | P2 — same stride model, checkpointed to OPFS |
| `Integration_timestep`, `Temperature`, `Pressure`, `Write_the_trajectory`, `Write_the_log` | as equilibration | | | P2 |
| *(implicit)* | | | builds a **fresh** `System` with **no restraints**, adds the barostat | P2 |
| *(implicit)* | | | sets `simulation.currentStep` and `context.setTime` by hand so logs continue across strides | P2 |

## Analysis (cells 27–29, 33–46)

| Field | Default | Options | What it does | Web app |
| --- | --- | --- | --- | --- |
| `Skip` | `1` | 1, 2, 5, 10, 20, 50 | frame stride when concatenating | P3 |
| `Output_format` | `dcd` | `dcd`, `pdb`, `trr`, `xtc` | MDAnalysis writer | P3 — `dcd` and `pdb` now; `trr`/`xtc` later or dropped (GROMACS formats, compressed, no reader needed by this pipeline) |
| `first_stride`, `Number_of_strides`, `stride_time`, `trajectory_saved_frequency` | `1`,`1`,`1`,`10` | | rebuild the time axis for the plots | P3 — taken from `run.json` instead of retyped |
| `Remove_waters` | `yes` | `yes`, `no` | writes `_nw.dcd` + `_nw.pdb` or `_whole.*` | P3 |
| *(implicit)* | | | `align.AlignTraj(u1, u2, select='name CA', in_memory=True)` against frame 0 of the first stride | P3 |
| cell 28 — trajectory viewer | — | — | writes ≤ 10 PDB frames, `py3Dmol.addModelsAsFrames`, `animate` | P3 — 3Dmol.js, all frames, play/scrub |
| cell 29 — `Output_name`, `Threshold` | `Interaction`, `0.3` | slider 0–1 | ProLIF fingerprint over ≤ 10 frames, `LigNetwork(kind="aggregate", threshold=…)` → HTML | P4 — 🔴 the notebook uses ≤ 10 frames here regardless of trajectory length (`stride_animation`), which makes a 0.3 frequency threshold coarse; the app uses every frame and says how many |
| cell 33 — `Frame` | `42` | string | writes `protein_ligand_{frame}.pdb` (no water/ions) | P4 — kept; the GNINA score print is dropped |
| cell 34 — ligand RMSD | `rmsd_lig` | | `MDAnalysis.analysis.rms.RMSD(select="resname UNK and not (name H*)")` against frame 0 | P4 — 🔴 depends on the ligand residue being named `UNK`; the app tracks the ligand by atom index |
| cell 35 | `rmsd_dist` | | seaborn KDE of the above | P4 |
| cell 36 — ligand/pocket distance | `distance`, `Distance=4` Å | | pytraj: residues with any atom within 4 Å of the ligand **in the first frame**, then `pt.distance` between the ligand mask and those residues (centre of mass), mean ± sd | P4 |
| cell 37 — named residues | `distance_select`, `Residues='78,84'` | | same, for residue numbers the user types | P4 |
| cell 38 — Cα RMSD | `rmsd_ca` | | `pt.rmsd(traj, ref=0, mask='@CA')` | P3 |
| cell 39 | `rmsd_dist` | | KDE | P3 |
| cell 40 — radius of gyration | `radius_gyration` | | `pt.radgyr(mask='@CA')` | P3 — 🔴 its time axis is built from `len(rmsd)`, not `len(radgyr)` |
| cell 41 | `radius_gyration_dist` | | KDE | P3 |
| cell 42 — Cα RMSF | `rmsf_ca` | | `pt.rmsf('@CA')`; also computes `pt.bfactors(byres=True)` and never uses it | P3 — the unused B-factor call is dropped |
| cell 43 — 2D RMSD | `2D_rmsd` | | `pt.pairwise_rmsd(mask='@CA')`, `PRGn`, bicubic | P3 |
| cell 44 — PCA | `PCA`, `PC1`, `PC2` | | `pt.pca(fit=True, ref=0, mask='@CA', n_vecs=2)`, scatter coloured by time | P3 |
| cell 45 | `PCA_dist` | | KDE of PC1 and PC2 | P3 |
| cell 46 — cross-correlation | `cross_correlation` | | `pt.align(mask='@CA', ref=0)` then `pytraj.matrix.correl('@CA')`, `PiYG_r`, −1…1 | P3 |
| *(implicit, all analyses)* | | | PNG at `dpi=600` and a `pandas` CSV per analysis | P3/P4 |

## Outputs

| Notebook file | Web app |
| --- | --- |
| `system.pdb` | P3 |
| `system_amber.prmtop`, `system_amber.inpcrd` | P3/P4 |
| `system_gromacs.top`, `system_gromacs.gro` | P5, best effort |
| `{job}.dcd`, `.log`, `.rst`, `.pdb` (equilibration) | P2 |
| `{job}_{n}.dcd`, `.log`, `.rst`, `.pdb` (per stride) | P2 |
| `{job}{range}_nw.dcd` / `_whole.dcd`, `_nw.pdb` / `_whole.pdb` | P3 |
| `initial.html`, `Interaction.html` (LigNetwork) | P4 |
| analysis `*.png` + `*.csv` | P3/P4 |
| — | **new:** `system.xml`, `state.xml` (the engine contract), `run.json` (every parameter, versions, seed, adapter, timings), and a "download all" zip |

---

## Measured differences between the notebook's labels and its behaviour

These are not criticisms of the notebook; they are the places where a faithful
port has to choose between copying the code and copying the stated intent, so
each one needs a decision. None of them is changed silently: whatever is
chosen goes in `docs/DEVIATIONS.md` and is shown in the UI when it affects a
run.

### 1. The `pH` field in cell 17 cannot change protein protonation 🔴

Cell 7 calls `fixer.addMissingHydrogens(pH=7.4)` with 7.4 written into the
source. By the time cell 17 calls `modeller.addHydrogens(forcefield, pH=float(pH))`
the protein already has every hydrogen, and the ligand arrives from an SDF with
explicit hydrogens, so there is nothing left to add.

Measured on 3HTB chain A + JZ4: `atoms_before_addHydrogens = 2636`,
`atoms_after_addHydrogens = 2636` — `addHydrogens` added nothing
(`measurements/3HTB-JZ4-facts.json`).

Consequence: setting `pH = 5` in cell 17 changes nothing. The web app has one
pH field and applies it where it actually decides protonation — the
PDBFixer/`Modeller` hydrogen step — and says so.

### 2. "Remove hydrogens" uses the PDB element column

`ppdb.df['ATOM']['element_symbol'] != 'H'` relies on columns 77–78 being
populated. Most RCSB entries fill them; hand-made and some converted files do
not, and those hydrogens survive into PDBFixer. The app detects a missing
element column and falls back to the atom-name rule, and reports which rule it
used.

Alternate locations are also worth naming: the biopandas filter keeps every
altLoc, and `openmm.app.PDBFile` then silently keeps only the first. 3HTB chain
A has 1236 atoms with no altLoc plus 64 `A` and 64 `B`, so 1364 records become
1300 atoms. The app shows the altLoc choice rather than making it invisibly.

### 3. The protein force-field labels do not match the files 🔴

| Notebook label | File it loads | What that file is |
| --- | --- | --- |
| `AMBER14SB` | `amber14/protein.ff14SB.xml` | ff14SB ✓ |
| `AMBER99SB` | `amber99sbildn.xml` | **ff99SB-ILDN** |
| `AMBER99` | `amber99sb.xml` | **ff99SB** |

So the option called `AMBER99` gives ff99SB and the one called `AMBER99SB`
gives ff99SB-ILDN. The UI uses ff14SB / ff99SB / ff99SB-ILDN and names the file
each one loads.

### 4. Padding is the total added width, and the default lets the solute see its own image 🔴

Two separate problems. First, what padding means. `Modeller.addSolvent`
(openmm/app/modeller.py:499-512) computes

```python
radius = max(norm(center - pos) for pos in positions)   # bounding SPHERE
width  = max(2*radius + padding, 2*padding)
```

so the padding is the **total** extra width — padding/2 of clearance per side —
and it is measured from a bounding sphere, not from the per-axis extent. The
notebook's markdown says the opposite twice: "creates a cubic box of width
(solute size)+2*(padding)" and "guarantees that no part of the solute comes
closer than 1 nm to any edge of the box". At the 4 Å default the guaranteed
clearance is 2 Å per side.

Second, what that costs. The quantity that matters is not the padding but the
distance from a solute atom to the nearest image of the solute, against the
1.0 nm cutoff. Measured (`measurements/preparation-determinism.json`):

| system | box | nearest solute-to-own-image | inside the 1.0 nm cutoff? |
| --- | ---: | ---: | --- |
| 1L2Y (Trp-cage, 304 solute atoms) | 3.1127 nm | **0.9579 nm** | **yes** |
| 3HTB + JZ4 (2636 solute atoms) | 6.5239 nm | 1.8221 nm | no |

So the notebook's own quick test case, at the notebook's own defaults, has the
peptide interacting with its periodic image through the nonbonded cutoff.

The app keeps the 1–10 Å slider and the 4 Å default so runs stay comparable,
states the box semantics correctly, and after solvation computes the real
minimum image distance (a KD-tree over the solute, milliseconds) and warns when
it is below the cutoff. `padding < cutoff` stays as the cheap hint shown before
solvation.

### 5. The +30 Å shift is cosmetic

`positions = modeller.getPositions() + [30,30,30] Å`, with the comment "The box
is about 75 angstroms per side, so add (30,30,30) to center the protein". The
box is whatever the padding produced (65.2 Å for 3HTB), and under periodic
boundaries a rigid translation changes no energy. It does mean `system.pdb`
coordinates are offset from the input structure. The app does not shift, and
centres the view instead.

### 6. `LangevinIntegrator` is LFMiddle, not the leapfrog scheme 🔴

OpenMM 8.2 made `LangevinIntegrator` a subclass of `LangevinMiddleIntegrator`:
*"It is identical to LangevinMiddleIntegrator. (It simply subclasses it without
adding or changing anything.) The two different classes exist only for
historical reasons."* Verified on both installs here — in OpenMM 7.7
`LangevinIntegrator`'s MRO is `(LangevinIntegrator, Integrator)`, and in 8.4.0.dev
it is `(LangevinIntegrator, LangevinMiddleIntegrator, Integrator)`.

The notebook installs current OpenMM, so **it runs the LFMiddle
discretisation**, and the engine must implement LFMiddle to match it. The
original leapfrog Langevin is reachable only through a `CustomIntegrator` now.

### 7. The equilibration restraints are not heavy-atom restraints 🔴

The comment says "positional restraints for all heavy-atoms" and the mask is

```
!(:H*) & !(:HOH) & !(:NA) & !(:CL) & !(:K)
```

In a cpptraj mask `:` selects **residue** names and `@` selects atom names, so
`:H*` excludes residues whose name begins with H — `HOH`, and also every
histidine — while every hydrogen of every other residue stays in.

Measured on the 3HTB system (`measurements/restraint-mask.json`):

| mask | atoms restrained | of which hydrogen | histidine restrained |
| --- | ---: | ---: | --- |
| `!(:H*) & …` (as written) | **2619** | **1318** | **no** |
| `!(@H*) & …` (as intended) | **1311** | 0 | yes |

So the notebook restrains twice as many atoms as intended, including hydrogens,
and leaves histidine free. Restraining a hydrogen whose X–H bond is already
constrained is close to harmless; leaving histidine unrestrained while the rest
of the protein is held at 700 kJ/mol/nm² is not, and it scales with the
histidine content of the structure (one of 163 residues in 3HTB, far more in a
His-tagged construct).

**Decided (2026-10-19): offer both, defaulting to heavy atoms.** The app's
default is the heavy-atom mask `!(@H*) & !(:HOH) & !(:NA) & !(:CL) & !(:K)`,
which is what the comment, the protocol and the Making it Rain papers describe.
The notebook's own mask stays selectable as *"Making it Rain notebook
(hydrogens restrained, histidine free)"*, so a run can still be made
comparable with anything already published from the notebook. Which one was
used goes in `run.json` and on the equilibration panel. Recorded in
[DEVIATIONS.md](DEVIATIONS.md).

### 8. The PME grid depends on the platform, so "the same grid" is a choice

`calcPMEParameters` derives α and a *minimum* grid from `ewaldErrorTolerance`:

```cpp
alpha = (1.0/cutoff)*sqrt(-log(2.0*tol));
xsize = ceil(2*alpha*boxVector[0][0]/(3*pow(tol, 0.2)));   // ≥ 6
```

With `tol = 5e-4` and a 1.0 nm cutoff that is α = 2.628261 nm⁻¹ for every
system, and for 3HTB's 6.5239 nm box a minimum of 53. The Reference platform
uses 53 exactly; the CUDA and OpenCL platforms round up to a size whose prime
factors are small enough for their FFT. So the notebook's own grid depends on
which platform it ran on, and the engine's choice is a documented decision, not
a reproduction. The plan derives α by the formula above and rounds the grid up
to the nearest 2/3/5/7-smooth size, records both in `run.json`, and lets a run
pin an explicit grid for comparison.

### 9. Production depends on live notebook state

Cell 26 rebuilds the `System` from `modeller.topology` and `omm_forcefield`,
which exist only in the running kernel. Restarting the runtime and running only
the production cell cannot work, even though the `.rst` file is on disk. The web
app's `system.xml` + `state.xml` are the whole input to a production run, which
is what makes stop/resume and "upload and run" possible at all.

### 10. Minimisation runs with the restraints on

`restraints(...)` is applied before `minimizeEnergy`, so the 10000-step
minimisation is restrained, not free. Ported as-is; the UI says so.

### 11. Analysis time axes are rebuilt from retyped parameters

Cells 27–46 rebuild the time axis from `Write_the_trajectory`, `Skip`,
`stride_time` and `Number_of_strides` as retyped in the analysis cells; cell 40
uses `len(rmsd)` where it means `len(radgyr)`; the ligand network uses at most
10 frames whatever the trajectory length. The app reads all of this from
`run.json`, so the axis cannot disagree with the trajectory, and states the
frame count behind every plot.

### 12. The restraint force constant is labelled in the wrong units

`Force_constant = 700 #@param {type:"slider", min:0, max:2000, step:100}`, and
the comment on the line that reads it says `# kJ/mol`. It is not: `k` multiplies
`periodicdistance(...)^2`, so it is an energy per squared length, **kJ/mol/nm²**.
The printed line in the log repeats the wrong unit — `"Force Constant = 700
kJ/mol"`.

This one is only a label, and it is worth recording because the magnitude is
easy to get wrong in the other direction. 700 kJ/mol/nm² holds a heavy atom to
about `sqrt(kT/2k)` = 0.04 nm per axis at 300 K, which is a real restraint;
700 kJ/mol would be a number with no meaning here, and a reader who takes the
comment at face value and converts it has a restraint two orders of magnitude
off. Measured while building the restraint fixture: at 10 kJ/mol/nm² the
restrained atoms move indistinguishably from free ones.

The app shows the unit on the field and records `k` in `run.json` in
kJ/mol/nm².
