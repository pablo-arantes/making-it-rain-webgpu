# Making it Rain on WebGPU

Molecular dynamics in a browser tab, on your own GPU. Give it a PDB ID or a PDB
file, set the parameters you know from the
[Making it Rain](https://github.com/pablo-arantes/making-it-rain) Colab
notebooks, and run the simulation on your own machine. No account, no queue, no
server, and your structure is never uploaded.

> **Status: Phase 0.** There is no MD engine yet. What works today: the page
> reports your GPU and runs the precision checks the engine's design rests on,
> the five test lanes are wired up, and the Python oracle that the whole project
> is validated against regenerates its fixtures. `docs/PLAN.md` is the road map
> and `docs/PRECISION.md` is what Phase 0 settled.

## Read these first

| Document | What it is |
| --- | --- |
| [docs/PLAN.md](docs/PLAN.md) | what to build, in what order, and how each piece is held to the notebook |
| [docs/NOTEBOOK_PARITY.md](docs/NOTEBOOK_PARITY.md) | every field of `reference/Uni_Dock+MD.ipynb`, what it does, and where it lands |
| [docs/PRIOR_ART.md](docs/PRIOR_ART.md) | what already exists for this in a browser, its state, and its license |
| [docs/PRECISION.md](docs/PRECISION.md) | how forces and positions are represented without f64, and the two schemes that measurement eliminated |
| [docs/NONBONDED.md](docs/NONBONDED.md) | what OpenMM puts in which force group, and the three Ewald terms a first reading gets wrong |
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
