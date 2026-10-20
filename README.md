# Making it Rain on WebGPU

Molecular dynamics in a browser tab, on your own GPU. Give it a PDB ID or a PDB
file, set the parameters you know from the
[Making it Rain](https://github.com/pablo-arantes/making-it-rain) Colab
notebooks, and run the simulation on your own machine. No account, no queue, no
server, and your structure is never uploaded.

> **Status: Phase 2, experimental.** The engine runs. Upload an OpenMM
> `System` and `State` and the page minimises, equilibrates and produces by
> strides on your GPU, writing DCD trajectories, OpenMM-format logs and
> checkpoints you can take away. Every force, the constraints, both
> integrators, the thermostat, the barostat, the restraints and the minimiser
> are differentially tested against OpenMM. What is **not** built yet is the
> preparation path: Phase 3 turns a PDB ID into a solvated system, and until
> then you bring your own.

## What is measured, not claimed

On an Apple M3 Pro through Chrome 155, against OpenMM 8.4:

| | 1L2Y, 2,373 atoms | 3HTB + JZ4, 26,749 atoms |
| --- | --- | --- |
| energy per force group | 1.6e-6 to 8.1e-8 | 1.4e-7 to 1.7e-8 |
| direct-space forces | 6.9e-7 relative RMS | 5.4e-7 |
| one step vs OpenMM's own | 9.6e-8 nm | 1.0e-7 nm |
| energy drift (OpenMM's own) | 0.28 (0.37) | 0.54 (0.39) |
| throughput | 39 ns/day at 2 fs | 10 ns/day |
| two identical runs differ by | **0** | **0** |

Drift is kJ/mol per atom per ns from a least-squares slope, with OpenMM's figure
from the same configuration in brackets. Every number here is reproduced by
`npm run test:browser`, and `docs/measurements/` holds the commands.

## Read these first

| Document | What it is |
| --- | --- |
| [docs/PLAN.md](docs/PLAN.md) | what to build, in what order, and how each piece is held to the notebook |
| [docs/NOTEBOOK_PARITY.md](docs/NOTEBOOK_PARITY.md) | every field of `reference/Uni_Dock+MD.ipynb`, what it does, and where it lands |
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | what already exists for this in a browser, its state, and its license |
| [docs/PRECISION.md](docs/PRECISION.md) | how forces and positions are represented without f64, and the two schemes that measurement eliminated |
| [docs/NONBONDED.md](docs/NONBONDED.md) | what OpenMM puts in which force group, and the three Ewald terms a first reading gets wrong |
| [docs/DYNAMICS.md](docs/DYNAMICS.md) | the integrator and constraint algorithms, and why the NVE gate cannot use the notebook's integrator |
| [docs/MINIMISER.md](docs/MINIMISER.md) | what `LocalEnergyMinimizer` actually does, and where it stops short of its own tolerance |
| [docs/DEVIATIONS.md](docs/DEVIATIONS.md) | every place the app will differ from the notebook, and why |
| [AGENTS.md](AGENTS.md) · [CLAUDE.md](CLAUDE.md) | engineering invariants, and how to build and test this checkout |
| [docs/measurements/](docs/measurements/) | the numbers the documents quote, and the commands that produced them |

## Scope

Protein in water, and protein–ligand in water, with the notebook's protocol:
AMBER protein force fields, TIP3P or SPC/E water, PME at a 1.0 nm cutoff,
H-bond constraints, a Langevin (LFMiddle) integrator, a Monte Carlo barostat,
restrained equilibration, and production in strides — then the notebook's
analyses. The docking half of the reference notebook is out of scope.

## Acknowledgements

Making it Rain (Arantes, Polêto, Pedebos, Ligabue-Braun,
[JCIM 2021](https://doi.org/10.1021/acs.jcim.1c00998)) and
[Cloud-Bind](https://github.com/pablo-arantes/Cloud-Bind) are the protocol this
port follows. OpenMM, OpenFF, PDBFixer, RDKit, the AMBER force fields, 3Dmol.js
and ProLIF are the work this depends on;
[AlphaFold2 WebGPU](https://github.com/martin-steinegger/alphafold2-webgpu),
[LocalFold](https://github.com/sokrypton/localfold) and
[Molarium](https://github.com/rafwiewiora/molarium) showed that this class of
computation belongs in a browser. Full attribution will live in
`THIRD_PARTY.md`.

App code will be MIT, as Making it Rain is.
