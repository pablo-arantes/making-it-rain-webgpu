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
