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
