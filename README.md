# Making it Rain on WebGPU

Molecular dynamics in a browser tab, on your own GPU. Give it a PDB ID or a PDB
file, set the parameters you know from the
[Making it Rain](https://github.com/pablo-arantes/making-it-rain) Colab
notebooks, and run the simulation on your own machine. No account, no queue, no
server, and your structure is never uploaded.

> **Status: the whole chain works, and it is experimental.** Give the page a
> PDB ID or a PDB file and it prepares a solvated system in the browser —
> hydrogens at a pH you choose, a water box, counter-ions, AMBER parameters —
> then minimises, equilibrates and runs production in strides on your GPU,
> writing DCD trajectories, OpenMM-format logs, checkpoints and the notebook's
> analyses. Protein and protein–ligand both work, as do cubic, truncated
> octahedral and rhombic dodecahedral cells. You can still bring your own
> OpenMM `System` and `State` instead.
>
> Every force, the constraints, both integrators, the thermostat, the barostat,
> the restraints, the minimiser, the solvation and the trajectory writers are
> differentially tested against OpenMM — and OpenMM reads the `system.xml` this
> writes and agrees on the energy to 9.2e-12.
>
> **What is not done**, in the order it would bite you: the kernels have only
> ever run on Apple hardware and on a software adapter, never on an NVIDIA or
> Intel GPU; the acceptance thresholds are calibrated on one fixture through
> OpenMM's OpenCL platform rather than CUDA; the in-browser OpenMM oracle is
> not built, so parity is re-derivable only by running Python locally; and
> `docs/DEVIATIONS.md` lists 23 places this deliberately differs from the
> notebook, each shown to you in the page where it matters.

## What is measured, not claimed

On an Apple M3 Pro through Chrome 155, against OpenMM 8.4:

| | 1L2Y, 2,373 atoms | 3HTB + JZ4, 26,749 atoms |
| --- | --- | --- |
| energy per force group | 1.6e-6 to 5.4e-9 | 1.4e-7 to 1.7e-8 |
| direct-space forces | 6.9e-7 relative RMS | 5.4e-7 |
| one step vs OpenMM's own | 9.6e-8 nm | 1.0e-7 nm |
| energy drift (OpenMM's own) | 0.28 (0.40) | 0.54 (0.36) |
| throughput | 203 ns/day at 2 fs | 21 ns/day |
| two identical runs differ by | **0** | **0** |

A nanosecond end to end through the page, including minimisation and writing
the artefacts: **7.6 minutes** for 1L2Y, and **15.3 minutes** for a
5,002-atom rhombic dodecahedron — 2.11× the particles for 2.01× the time, so a
skewed cell costs nothing beyond the atoms it adds.

Against our own f64 reference on an equilibrated system the engine's forces are
**1.09e-6** relative RMS, which is **13× inside** OpenMM's own OpenCL
single-precision platform (1.43e-5) and 1.06× outside its CPU double. From a
PDB ID, the prepared system agrees with OpenMM to 9.2e-12 on the energy, and
the water this places is the water `Modeller.addSolvent` places — for a
dodecahedron, 483 of 483 oxygens within 1e-12 nm.

Drift is kJ/mol per atom per ns from a least-squares slope over ten reports,
with OpenMM's own figure from the same configuration and integrator in
brackets — it is a property of the integrator and the timestep rather than a
defect, which is why ours is better than OpenMM's on one fixture and worse on
the other.

🔴 **Both sides of that row are stochastic.** They start from assigned
velocities, and repeated runs here moved the same figure by tens of percent —
3HTB read 0.35 and 0.46 on two runs an hour apart. So it is one calibrated pair
and not a bound: the regression gate is **1.5**, a little under three times the
worse measurement, and two significant figures of either number should not be
read as a claim about the floor. Ours comes from the calibration in
`src/gpu/validate-loop.ts`; OpenMM's from
`docs/measurements/drift-openmm-cpu-{1L2Y,3HTB}.json`.

Everything else here is reproduced by `npm run test:browser`, and
`docs/measurements/` holds the commands and the raw output.

🔴 Measured on **one** machine, an Apple M3 Pro through Chrome 155. Nothing here
has been confirmed on another GPU.

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
computation belongs in a browser. [THIRD_PARTY.md](THIRD_PARTY.md) has the full
attribution, including the licence of every redistributed file.

App code is MIT, as Making it Rain is — see [LICENSE](LICENSE).
