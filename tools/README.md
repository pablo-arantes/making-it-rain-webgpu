# tools

The Python side: the oracle that every browser-built system and every WGSL
kernel is checked against, and the probes whose numbers the documents in `docs/`
quote.

## Environment

```sh
conda env create -f tools/environment.yml
conda activate mir-oracle
```

`sqm` and `antechamber` must be on `PATH` for any `gaff-*` ligand force field;
they come with `ambertools`. Everything else runs without them.

## The oracle

```sh
# protein only, the quick case
python tools/build_reference_system.py --pdb-id 1L2Y --out fixtures/

# the notebook's default protein-ligand case
python tools/build_reference_system.py --pdb-id 3HTB --ligand JZ4 --out fixtures/

# the same with OpenFF instead of GAFF
python tools/build_reference_system.py --pdb-id 3HTB --ligand JZ4 \
    --ligand-ff openff-2.0.0 --case 3HTB-openff --out fixtures/
```

Writes `receptor.pdb`, `ligand.sdf`, `system.pdb`, `system.xml`, `state.xml` and
`facts.json` per case. `facts.json` is small and committed under
`docs/measurements/`; the rest is regenerated, never committed.

This runs the topology half of `reference/Uni_Dock+MD.ipynb` — the receptor
preparation of cell 7 and the whole of cell 17 — with the docking removed and
the ligand taken from a HETATM residue of the PDB entry, with bond orders from
the RCSB Chemical Component Dictionary. That is the route the browser takes, so
the two are comparable by construction.

## Parameter parity fixtures

Phase 3 reimplements OpenMM's parameter assignment, and the gate is term-by-term
parity rather than an energy — an energy is a sum, and a wrong torsion and a
wrong charge can cancel inside a tolerance.

```sh
# a small protein with its hydrogens, as OpenMM's own PDB reader sees it
python tools/build_prep_fixture.py --pdb-id 1L2Y --out fixtures/prep/

# every parameter OpenMM assigned: masses, charges, bonds, angles, torsions
python tools/dump_parameters.py --pdb fixtures/prep/1L2Y.pdb \
    --forcefield amber14/protein.ff14SB.xml --out fixtures/prep/1L2Y-params.json

# which template each residue matched, and the atom correspondence inside it
python tools/dump_template_match.py --pdb fixtures/prep/1L2Y.pdb \
    --forcefield amber14/protein.ff14SB.xml --out fixtures/prep/1L2Y-match.json

# and the same three for the second fixture, T4 lysozyme
python tools/build_prep_fixture.py --pdb fixtures/3HTB/receptor.pdb \
    --out fixtures/prep/3HTB
python tools/dump_parameters.py --pdb fixtures/prep/3HTB.pdb \
    --forcefield amber14/protein.ff14SB.xml --out fixtures/prep/3HTB-params.json
python tools/dump_template_match.py --pdb fixtures/prep/3HTB.pdb \
    --forcefield amber14/protein.ff14SB.xml --out fixtures/prep/3HTB-match.json
```

```sh
# the configuration the notebook actually runs: protein, water and ions
cp fixtures/1L2Y/system.pdb fixtures/prep/1L2Y-solv.pdb
python tools/dump_parameters.py --pdb fixtures/prep/1L2Y-solv.pdb \
    --forcefield amber14/protein.ff14SB.xml amber14/tip3p.xml \
    --out fixtures/prep/1L2Y-solv-params.json \
    --system-out fixtures/prep/1L2Y-solv-system.xml
python tools/dump_template_match.py --pdb fixtures/prep/1L2Y-solv.pdb \
    --forcefield amber14/protein.ff14SB.xml amber14/tip3p.xml \
    --out fixtures/prep/1L2Y-solv-match.json
```

The solvated case is the only one that exercises `mergeForceFields`, the `HOH`
template and the monovalent ion templates — and it is what caught the wrong
`tip3p.xml` being vendored: OpenMM's top-level `tip3p.xml` holds water alone,
while `amber14/tip3p.xml` carries the ions, so a system with two Na+ and three
Cl- could not be prepared at all.

Pass `--system-out fixtures/prep/<name>-system.xml` to `dump_parameters.py` as
well, which serialises the `System` itself. That file is what
`test/prep-system.test.ts` compares against, and it is a stronger test than the
JSON dump: the engine contract *is* `System` XML, so comparing against the file
the engine consumes catches anything a summary of it would smooth over — the
bond ordering did exactly that.

1L2Y is 20 residues with no histidine, no cysteine and no sulfur, so it cannot
tell a rule that is right from one that is merely right for it. 3HTB's receptor
is 163 residues and 2614 atoms with all twenty standard residues, and adds the
cases that matter: HID against HIE, which share an element signature and are
separated only by which ring nitrogen carries the proton, plus methionine's
sulfur and 8842 torsions.

The second dump exists because charges cannot distinguish two valid graph
isomorphisms of a symmetric residue — tyrosine's CD1 and CD2 carry identical
charges — while `ordering="amber"` breaks improper torsion ties on the atom's
index *inside its template*. So the choice is invisible in the charges and
visible in the energy. It also records `bondedToAtom` in OpenMM's own order,
because the improper match cache is keyed by type signature and replays
positions, which makes candidate order reach the result.

All three are regenerated, never committed; `test/prep-*.test.ts` skip
themselves and say so when they are absent.

## Hydrogens

```sh
# what Modeller.addHydrogens produced: variants, atoms, bonds
python tools/dump_hydrogens.py --pdb fixtures/1L2Y/receptor.pdb \
    --out fixtures/prep/1L2Y-hydrogens.json
python tools/dump_hydrogens.py --pdb fixtures/3HTB/receptor.pdb \
    --out fixtures/prep/3HTB-hydrogens.json
```

Hydrogens and water are stripped first, so the comparison is against what
OpenMM builds from heavy atoms rather than against whatever the depositor named
their hydrogens.

The dump records no positions, deliberately. OpenMM places each hydrogen at its
parent plus a `random.random()` offset and then minimises the set, so it does
not reproduce its own hydrogen positions between runs; there is nothing there to
compare. See deviations 16 and 17 in `docs/DEVIATIONS.md` — the second is that
OpenMM's documented HID/HIE hydrogen-bond rule is unreachable code and always
yields HID.

## Solvation

```sh
python tools/dump_solvent.py --pdb fixtures/prep/1L2Y.pdb \
    --out fixtures/prep/1L2Y-solvent.json
```

Runs `addSolvent` twice, because only part of it is reproducible. With ions off,
the water placement is deterministic and every molecule is comparable; with the
notebook's 0.15 M NaCl, only the counts are, because `_addIons` picks the waters
to replace with `random.choice`. Deviations 18 and 19 in `docs/DEVIATIONS.md`.

The pre-equilibrated water boxes live in `vendor/water/` (`tip3p.pdb`,
`spce.pdb`, 3 nm cubes of 895 molecules each) and are recorded in
`THIRD_PARTY.md`.

## Ligands (Phase 4)

A PDB does not say what a ligand *is*: `HETATM` records carry an element, a name
and coordinates, and no bonds. The chemistry comes from the RCSB Chemical
Component Dictionary.

```sh
# the component definition and the ideal coordinates, per three-letter code
curl -o fixtures/ligand/JZ4.cif https://files.rcsb.org/ligands/download/JZ4.cif

# one ligand on its own, which is what RDKit needs
python tools/extract_ligand.py --pdb fixtures/ligand/181L.pdb --code BNZ \
    --out fixtures/ligand/BNZ_fragment.pdb

# RDKit's perception, as the reference
~/miniforge3/envs/parametrizani_env/bin/python tools/dump_ligand.py \
    --cif fixtures/ligand/JZ4.cif --fragment fixtures/3HTB/ligand_fragment.pdb \
    --ideal fixtures/3HTB/JZ4_ideal.sdf --out fixtures/ligand/JZ4.json
```

🔴 The claim being tested is that **transferring bond orders by atom name gives
the same answer as RDKit's `AssignBondOrdersFromTemplate`**, which is a
substructure match the browser cannot run. The names in a deposited `HETATM`
record are the CCD's own `atom_id` values, so the correspondence is already in
the file. Verified on both ligands the plan's gate names — JZ4 from 3HTB and
benzene from 181L — bond for bond including aromatic flags, with hydrogen counts
derived from valence matching RDKit exactly.

Name-keyed transfer is also *safer* than a substructure match, not merely
simpler: a match has to choose among symmetry-equivalent mappings, and benzene
has six equally good ones. The orders come out right either way; which atom
they attach to does not.

## Amber export

```sh
npx tsx tools/export_prepared.ts fixtures/1L2Y/receptor.pdb fixtures/crossfeed
# then write prmtop/inpcrd from any System this project holds, and check them
python tools/check_prmtop.py --prmtop fixtures/amber/1L2Y.prmtop \
    --inpcrd fixtures/amber/1L2Y.inpcrd \
    --system fixtures/prep/1L2Y-system.xml --out fixtures/amber/1L2Y-energies.json
```

🔴 "Does it load" is not a test of this format. A prmtop with an exclusion
window off by one, a double-counted 1-4, or a transposed Lennard-Jones type
matrix parses perfectly and gives a plausible number. `check_prmtop.py` has
OpenMM build a `System` from the file and compares per-force-group energies
against the `System` the file was written from. Agreement: 5.2e-9 relative on
the nonbonded group, 1.3e-8 on the total, with the residual being the kcal/mol
round trip.

That comparison caught the one real bug. `assignPropers` drops torsion terms
with a zero force constant because OpenMM does — but OpenMM still creates the
**1-4 exception** for the pair they span, and Amber has nowhere else to put
one. 184 of 1L2Y's 812 scaled 1-4 pairs are spanned only by zero-constant
torsions, so without synthetic dihedrals to carry them the nonbonded energy was
out by 2791 kJ/mol against a −2431 kJ/mol true value, while every bonded term
still matched to 1e-10.

## Cross-fed energies

The plan's Phase 3 gate, and the one the term-by-term comparison does not
replace: parameter parity shows the two engines assign the same numbers to the
same terms, not that they agree on an energy.

```sh
# prepare a system here and write it out as OpenMM's own serialisation
npx tsx tools/export_prepared.ts fixtures/1L2Y/receptor.pdb fixtures/crossfeed

# have OpenMM read that file and evaluate it, per force group
python tools/cross_feed_energy.py --system fixtures/crossfeed/system.xml \
    --state fixtures/crossfeed/state.xml --out fixtures/crossfeed/energies.json
```

The system is deliberately the one with **no OpenMM counterpart**: 1L2Y with
hydrogens this code placed and a water box this code filled, which OpenMM never
built and so cannot be compared structurally.

Measured on a 2400-atom box, against double-precision Reference:

| force group | relative difference |
| --- | --- |
| HarmonicBondForce | 1.7e-16 |
| HarmonicAngleForce | 8.8e-16 |
| PeriodicTorsionForce | 1.6e-16 |
| NonbondedForce (direct + reciprocal) | 5.1e-12 |
| total | 9.2e-12 |
| worst sampled force component | 5.8e-9 kJ/mol/nm |

Three things make that comparison fair rather than approximate. Each force goes
in its own group, because a total is a sum and a bonded term too high cancels a
nonbonded term too low. The PME mesh is the one OpenMM actually used, asked for
through `getPMEParametersInContext` — the Reference platform takes the exact
minimum grid while the engine rounds up to a smooth FFT size, and comparing
those compares two different meshes. And the exported system has no
constraints, since a constrained bond is removed from the `HarmonicBondForce`
and the two systems would have genuinely different bonded energies.

## Ligand typing by uploaded parameters (Phase 4)

The route that makes browser-side protein–ligand preparation possible without
RDKit in WebAssembly. `openmmforcefields` emits an ffxml holding a residue
template plus types, bonded terms and nonbonded parameters, and
`src/prep/ffxml.ts` already reads exactly that — so the ligand is parameterised
once, anywhere, and the browser merges the result.

```sh
~/miniforge3/envs/parametrizani_env/bin/python tools/dump_ligand_ffxml.py \
    --sdf fixtures/3HTB/ligand.sdf --method smirnoff \
    --out fixtures/ligand/JZ4-ligand.xml

# then the usual parity dumps, with the ligand ffxml as a third force field
python tools/dump_parameters.py --pdb fixtures/3HTB/system.pdb \
    --forcefield amber14/protein.ff14SB.xml amber14/tip3p.xml \
    fixtures/ligand/JZ4-ligand.xml \
    --out fixtures/prep/3HTB-lig-params.json \
    --system-out fixtures/prep/3HTB-lig-system.xml
```

**Charges come from NAGL, not AM1-BCC**, and not only because AmberTools' `sqm`
is unavailable here: the browser cannot run `sqm` either, so NAGL is the method
the engine will actually use (deviation 6). Using it on both sides makes the
comparison measure the *typing* rather than the charge method.
`openff-gnn-am1bcc-1.0.0.pt` is the model, and it gives a total charge of
−0.000000 e on JZ4.

Result: every field of every force matches OpenMM element for element on the
full 26,749-atom solvated system — 18698 bonds, 12809 angles, 8910 torsions,
38401 exceptions, and all 26749 masses, charges, sigmas and epsilons. The
ligand is matched by **graph**, which is what makes this work at all: the
template is named after the molecule's SMILES and its atoms carry Tripos
names, and neither appears in the PDB.

🔴 One real bug, worth 2e-11 and easy to dismiss as round-off. A SMIRNOFF
ligand ffxml declares `coulomb14scale="0.8333333333"` where ff14SB declares
`0.8333333333333334`. `mergeForceFields` took the *last* file's value, so the
ligand's coarser scale was applied to every **protein** 1-4 pair, shifting all
38401 exception charge products. OpenMM keeps the **first** and raises past
`SCALETOL = 1e-5`; so does this now.

## Ligand analyses (Phase 4)

```sh
# a protein-ligand trajectory: 3HTB with its JZ4 ligand, present as UNK
python tools/build_analysis_fixture.py --system fixtures/3HTB/system.xml \
    --state fixtures/3HTB/state.xml --pdb fixtures/3HTB/system.pdb \
    --frames 60 --interval 100 --out fixtures/analysis-3HTB

# MDAnalysis and ProLIF, in the one env that has both
~/miniforge3/envs/prolif/bin/python tools/dump_ligand_analyses.py \
    --dcd fixtures/analysis-3HTB/trajectory.dcd --pdb fixtures/3HTB/system.pdb \
    --ligand UNK --out fixtures/analysis-3HTB/ligand.json
```

Cells 33 to 37. Agreement: ligand RMSD 1.2e-7 nm, centre of mass 9.8e-8 nm,
pocket residues identical, pocket and named-residue distances 1.4e-7 nm.

🔴 `build_analysis_fixture.py` now minimises before integrating. Without it
3HTB reaches `Particle coordinate is nan` within a few hundred steps of 2 fs
Langevin from the raw solvated coordinates — 1L2Y happened to survive, which is
not a property to depend on.

🔴 Two findings, deviations 22 and 23. The notebook's ligand RMSD superposes the
selection first, so it measures the ligand's **conformation**, not its
displacement — under 0.032 nm here while the centre of mass moves further. And
`resname UNK` selects nothing on a system prepared outside the notebook, which
MDAnalysis reports as an RMSD of zero rather than an error.

**The interaction network is not done.** ProLIF on this trajectory returns only
three `VdWContact` interactions, because a ligand read from a PDB has no bond
orders and ProLIF cannot perceive it properly. A useful parity test needs the
ligand supplied with its chemistry — which `src/prep/ccd.ts` can now provide —
so the network waits on wiring those together rather than on the fingerprint
itself.

## Protein analyses

```sh
python tools/build_analysis_fixture.py --system fixtures/1L2Y/system.xml \
    --state fixtures/1L2Y/state.xml --pdb fixtures/1L2Y/system.pdb \
    --out fixtures/analysis
```

Runs 60 frames of Langevin dynamics and computes every analysis independently:
the RMSD with **MDTraj and** a NumPy SVD Kabsch fit, the rest from the formulae
written out in the tool. Two superposition algorithms agreeing is evidence; one
algorithm compared with itself is not.

The plan's gate names **pytraj**, and that gate is now met:

```sh
# pytraj needs an env where it imports; `ambertools` here is an x86_64 build
# that will not load on arm64, and `parametrizani_env` works
~/miniforge3/envs/parametrizani_env/bin/python tools/pytraj_analyses.py \
    --dcd fixtures/analysis/trajectory.dcd --pdb fixtures/1L2Y/system.pdb \
    --out fixtures/analysis/pytraj.json
```

| analysis | agreement with pytraj |
| --- | --- |
| `pt.rmsd` | 5.9e-8 nm |
| `pt.radgyr` | 4.8e-8 nm |
| `pt.rmsf` (on an aligned trajectory) | 1.9e-8 nm |
| `pt.pairwise_rmsd` | 7.3e-8 nm |
| `pt.pca` eigenvalues | 6.4e-10 nm² |
| `pt.pca` projections | exact, up to each eigenvector's sign |
| `pytraj.matrix.correl` | 7.7e-7 |

🔴 Two findings came out of running it, both recorded as deviations 20 and 21.
`pt.rmsf` does no fitting, so the notebook's cell 42 measures tumbling as well
as internal motion: on these frames the maximum is 0.0737 nm aligned against
0.1636 nm raw, a factor of 2.2. And cpptraj's PCA eigenvector signs are
whatever its solver produced — PC1 agrees in every frame here and PC2 is
flipped in every frame, which is the same component and a mirrored axis.

## Trajectory readers

```sh
# a synthetic fixture, against the arrays it was built from
python tools/check_dcd.py --dcd fixtures/dcd/ours.dcd --fixture fixtures/dcd

# a trajectory a real run produced, cross-read between readers
python tools/crossread_dcd.py --dcd fixtures/run-artefacts/equil.dcd \
    --pdb fixtures/run-artefacts/equil.pdb
```

Two halves of one gate. `check_dcd.py` compares against known inputs, which
needs a fixture. `crossread_dcd.py` asks whether independent readers agree about
a file where there is no reference — which is the only question available for a
real run, and the one the synthetic fixture cannot answer, because it is 37
atoms and one frame per axis. Header fields that happen to be right for a small
file stop being right at thousands of atoms over tens of frames.

Measured on `fixtures/run-artefacts/equil.dcd`, written by `web/run.html` from a
2373-particle 1L2Y run: MDAnalysis and MDTraj agree on 2373 atoms, the frame
count, a 3.0806 nm box, and coordinates to **4.77e-07 nm** — f32 round-off, the
file holding f32.

🔴 **pytraj is not installed in any environment on this machine**, so the plan's
gate is only partly met: `pytraj` is also what covers VMD, since cpptraj reads
DCD through VMD's own `dcdplugin`. Until it is available, "opens in VMD" is
unverified rather than passing.

## Probes

```sh
# what the notebook's restraint mask actually selects
python tools/probe_restraint_mask.py fixtures/3HTB/system.pdb \
    docs/measurements/restraint-mask.json

# is the preparation reproducible, and is the box big enough
python tools/probe_preparation_determinism.py fixtures/3HTB/receptor.pdb \
    --solvated fixtures/1L2Y/system.pdb --solvated fixtures/3HTB/system.pdb \
    --out docs/measurements/preparation-determinism.json

# Reference energies per force group, and ns/day on this host's platforms
python tools/bench_reference_host.py fixtures/3HTB/system.xml \
    fixtures/3HTB/state.xml --out docs/measurements/host-timings.json
```

## Still to write

* `calibrate_thresholds.py` — OpenMM CUDA mixed vs Reference on the same
  fixtures, to set the engine's acceptance thresholds instead of guessing them.
  Needs a CUDA machine; Colab is the intended home.
* `compare_charges.py` — NAGL against AmberTools AM1-BCC per atom.
* `capture_analysis_reference.py` — pytraj/MDAnalysis/ProLIF results on a
  committed trajectory, as the analysis gates.

## Threshold calibration on Colab

The engine's acceptance thresholds are "no worse than OpenMM mixed precision,
within a recorded factor". That needs a real GPU. On a Colab GPU runtime:

```python
!pip -q install condacolab
import condacolab; condacolab.install()          # restarts the kernel
```

```python
!mamba install -q -y -c conda-forge openmm
!git clone -q https://github.com/pablo-arantes/making-it-rain-webgpu
%cd making-it-rain-webgpu
# fixtures are regenerated, not committed; build them first or upload them
!python tools/calibrate_thresholds.py fixtures/3HTB/system.xml \
    fixtures/3HTB/state.xml --out docs/measurements/calibration-T4.json
```

It minimises and runs 20 ps of NVT before comparing, because at a fixture's raw
starting coordinates the relative force error reports on the input structure
rather than on the platform. Then, for every platform and precision the host
offers, against the double-precision Reference: per-force-group energies,
per-atom force error, NVE drift at 2 fs, and ns/day.

`proposed_thresholds` in the output is what goes into `AGENTS.md`, named with
the GPU that produced it. A run whose basis is not mixed precision says so and
is marked provisional — which is the state of the committed M3 Pro numbers,
since this Mac's `OpenCL/mixed` is unavailable and only `OpenCL/single` works.
