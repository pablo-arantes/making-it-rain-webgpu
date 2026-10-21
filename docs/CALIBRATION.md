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

🔴 **No GitHub needed, and no clone.** An earlier draft of this file told you to
`git clone https://github.com/pablo-arantes/making-it-rain-webgpu`. That repo
does not exist — this checkout has no remote configured and all 59 commits are
local — so that instruction could never have worked.

It is also unnecessary. `tools/calibrate_thresholds.py` and
`tools/build_reference_system.py` import nothing from this repository: only
`numpy`, `openmm`, `pdbfixer` and (for a ligand) `rdkit`. Together they are 29
kB. Upload those two files and you are done.

Six cells. The kernel restarts after the second one, which is normal.

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

A few minutes. `openmm` brings the CUDA platform; `pdbfixer` is what builds the
fixture.

### 4. Upload the two scripts

```python
from google.colab import files
files.upload()        # pick calibrate_thresholds.py and build_reference_system.py
```

They land in `/content`, which is where the next two cells run.

### 5. Build a fixture

Fixtures are regenerated rather than committed, so there is nothing to
calibrate against until this runs.

```python
!python build_reference_system.py --pdb-id 1L2Y --out fixtures/
```

The protein-only case: about 2,373 atoms, a couple of minutes, and it needs
nothing beyond `openmm` and `pdbfixer`. It downloads 1L2Y from the RCSB itself.

### 6. Calibrate

```python
!python calibrate_thresholds.py fixtures/1L2Y/system.xml \
    fixtures/1L2Y/state.xml --out calibration-T4.json
```

Then read the headline numbers and download the file:

```python
import json
print(json.dumps(json.load(open("calibration-T4.json"))["proposed_thresholds"], indent=1))
files.download("calibration-T4.json")
```

## 🔴 If `CUDA` appears but produces no row

This happened on the first attempt, and it is the failure worth knowing about
because the output looks finished. `docs/measurements/calibration-T4-incomplete.json`
is the record: `available_platforms` lists `Reference`, `CPU` and **`CUDA`**,
and `platforms` contains only `CPU`. No `proposed_thresholds` block was written,
which is correct — but nothing in the file said why.

**OpenMM registering a platform does not mean the platform works.** The CUDA
plugin is listed as soon as it loads, which needs only `libcuda` — not a
device, not a matching driver, not a usable toolkit. "CUDA is available" and "a
Context can be made on CUDA" are different claims.

Run this before cell 5 and it will name the cause:

```python
import openmm
from openmm import Platform
print("openmm", openmm.version.version)
print("plugin load failures:", Platform.getPluginLoadFailures() or "none")
print("platforms:", [Platform.getPlatform(i).getName()
                     for i in range(Platform.getNumPlatforms())])

one = openmm.System(); one.addParticle(1.0)
try:
    ctx = openmm.Context(one, openmm.VerletIntegrator(0.001),
                         Platform.getPlatformByName("CUDA"))
    print("CUDA works:", ctx.getPlatform().getPropertyValue(ctx, "DeviceName"))
except Exception as exc:
    print("CUDA cannot make a Context:", type(exc).__name__, exc)
```

`getPluginLoadFailures()` is the one that usually answers it — a missing
`libcufft` or `libcudart` shows up there by name.

The two causes in practice:

**No GPU attached.** `nvidia-smi` in cell 1 errors or lists nothing. Fix it in
**Runtime → Change runtime type → T4 GPU**. The CUDA platform can still be
registered on a CPU runtime, which is exactly why cell 1 exists.

**A toolkit the driver cannot run.** conda-forge's `openmm` is built against a
particular CUDA version, and the one `mamba` resolves may not match Colab's
driver. Pin it to what `nvidia-smi` reports — a 550-series driver is CUDA 12:

```python
!mamba install -q -y -c conda-forge openmm pdbfixer "cuda-version=12"
```

Then re-run the diagnostic. `calibrate_thresholds.py` now records every
candidate's failure under `skipped_platforms` and writes a `verdict` block
saying in plain words whether the run met its purpose, so a second attempt is
self-describing and you will not need to copy anything out of the notebook.

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

Send me the JSON. Its home in the repository is
`docs/measurements/calibration-T4.json`, and the `proposed_thresholds` block
goes into `AGENTS.md`, **named with the GPU that produced it**. That is the change that turns the
numbers in this repository from self-consistent into validated, and it lets the
provisional markers come out.

## Optional: the bigger case

3HTB with its ligand is the notebook's default and a better stress test at
~26,700 atoms, but it needs the heavier stack:

```python
!mamba install -q -y -c conda-forge openff-toolkit openmmforcefields rdkit parmed
!python build_reference_system.py --pdb-id 3HTB --ligand JZ4 --out fixtures/
!python calibrate_thresholds.py fixtures/3HTB/system.xml \
    fixtures/3HTB/state.xml --out calibration-T4-3HTB.json
```

🔴 Budget for this. The `CPU` row runs 25,000 NVE steps on 26,700 atoms, which
is tens of minutes on a Colab CPU and is the slowest part of the whole script by
a wide margin. `--skip-nve` drops the dynamics gate and keeps the single-point
and speed numbers if the session is running short; `--nve-steps 5000` keeps a
weaker version of it. The drift slope is fitted over the run, so fewer steps
means a noisier slope, not a wrong one.
