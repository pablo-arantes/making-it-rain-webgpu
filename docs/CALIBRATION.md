# Calibrating the acceptance thresholds on a Colab T4

## Why this is not optional

Every acceptance threshold in `AGENTS.md` is marked **provisional**, and the
reason is specific. The engine's target is not "agrees with double precision" —
nothing in single precision does. It is *"no worse than OpenMM's own
mixed-precision GPU platform, within a small factor"*. That reference number has
to be measured on a real GPU.

The development machine (M3 Pro, macOS 14) cannot produce it. Its
`OpenCL/mixed` platform is unavailable and only `OpenCL/single` works, so the
committed numbers are calibrated against the **CPU** platform instead — a
different precision class from the one the engine targets. Until this is run,
every "inside tolerance" claim in the repository means inside a tolerance that
nobody has validated against the right baseline.

A T4 is the right machine because it is also where the notebook runs.

## What to do

Six cells. The kernel restarts after the first one, which is normal.

### 1. Check you actually have a GPU

```python
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

Expect `Tesla T4`. If this errors, the runtime is CPU-only: **Runtime → Change
runtime type → T4 GPU**, then start again. A CPU runtime will still produce a
JSON file, and it will be worthless for this purpose — the whole point is the
`CUDA/mixed` row.

### 2. conda, because OpenMM needs it

```python
!pip -q install condacolab
import condacolab; condacolab.install()
```

**The kernel restarts here.** Wait for it, then carry on at cell 3 — do not
re-run cell 2.

### 3. OpenMM

```python
!mamba install -q -y -c conda-forge openmm pdbfixer
```

Takes a few minutes. `pdbfixer` is what builds the fixture; `openmm` brings the
CUDA platform.

### 4. The repository

```python
!git clone -q https://github.com/pablo-arantes/making-it-rain-webgpu
%cd making-it-rain-webgpu
```

### 5. Build a fixture

Fixtures are regenerated rather than committed, so there is nothing to
calibrate against until this runs.

```python
!python tools/build_reference_system.py --pdb-id 1L2Y --out fixtures/
```

That is the protein-only case: about 2,373 atoms, a couple of minutes, and it
needs nothing beyond `openmm` and `pdbfixer`.

### 6. Calibrate

```python
!python tools/calibrate_thresholds.py fixtures/1L2Y/system.xml \
    fixtures/1L2Y/state.xml --out docs/measurements/calibration-T4.json
```

Then read it back and download it:

```python
import json; print(json.dumps(
    json.load(open("docs/measurements/calibration-T4.json"))["proposed_thresholds"],
    indent=1))
from google.colab import files; files.download("docs/measurements/calibration-T4.json")
```

## What it should print

One line per platform and precision the runtime offers. On a T4 expect
`CUDA/double`, `CUDA/mixed`, `CUDA/single` and `CPU`; `OpenCL` may or may not
appear and `HIP` will not.

The row that matters is **`CUDA/mixed`** — that is the comparison class, and
`proposed_thresholds` is derived from it. Rough expectations, so that something
badly wrong is recognisable:

| quantity | plausible on a T4 |
| --- | --- |
| `CUDA/mixed` force relative RMS | around 1e-5 |
| `CUDA/double` force relative RMS | around 1e-10 or below |
| `CUDA/mixed` NVE drift | a few 1e-4 kJ/mol/ns/atom or less |
| `CUDA/mixed` speed, 1L2Y | tens to low hundreds of ns/day |

If `CUDA/mixed` reports a force relative RMS near 1e-10, the precision argument
is not selecting; if it reports 1e-2, something is wrong with the fixture rather
than the platform.

## Afterwards

Send me the JSON, or commit it — `docs/measurements/calibration-T4.json` is the
intended home — and the `proposed_thresholds` block goes into `AGENTS.md`,
**named with the GPU that produced it**. That is the change that turns the
numbers in this repository from self-consistent into validated, and it lets the
provisional markers come out.

## Optional: the bigger case

3HTB with its ligand is the notebook's default and a better stress test at
~26,700 atoms, but it needs the heavier stack:

```python
!mamba install -q -y -c conda-forge openff-toolkit openmmforcefields rdkit parmed
!python tools/build_reference_system.py --pdb-id 3HTB --ligand JZ4 --out fixtures/
!python tools/calibrate_thresholds.py fixtures/3HTB/system.xml \
    fixtures/3HTB/state.xml --out docs/measurements/calibration-T4-3HTB.json
```

🔴 Budget for this. The `CPU` row runs 25,000 NVE steps on 26,700 atoms, which
is tens of minutes on a Colab CPU and is the slowest part of the whole script by
a wide margin. `--skip-nve` drops the dynamics gate and keeps the single-point
and speed numbers if the session is running short; `--nve-steps 5000` keeps a
weaker version of it. The drift slope is fitted over the run, so fewer steps
means a noisier slope, not a wrong one.
