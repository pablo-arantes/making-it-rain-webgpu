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

🔴 The plan's gate names **pytraj**, which is not installed in any environment
here. So this shows the implementation matches the stated definitions and an
independent implementation of them — not that it matches *cpptraj's*
conventions. That matters where cpptraj is ambiguous: it does no alignment
inside `rmsf` or `matrix.correl`, so `src/analysis/protein.ts` says what it
does instead of inheriting an accident. Running this tool somewhere pytraj
exists would close the gap.

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
