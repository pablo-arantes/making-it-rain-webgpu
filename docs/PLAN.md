# Plan

Making it Rain on WebGPU: the notebook's protein and protein–ligand explicit-solvent
NPT protocol, run entirely in a browser tab, with no account, no queue, no
server, and no upload of anyone's structure.

Read [NOTEBOOK_PARITY.md](NOTEBOOK_PARITY.md) for what the notebook does and
[PRIOR_ART.md](PRIOR_ART.md) for what already exists. This file is what to
build, in what order, and how each piece is held to the notebook.

---

## Decisions at a glance

| Question | Decision | Why |
| --- | --- | --- |
| How does system building run in the browser? | **Hybrid: TypeScript reimplementation in production, real OpenMM (WebAssembly, Reference platform) as the oracle** — in CI, and behind a "verify this system" button in the UI | The Python stack is blocked in four independent places; the TS subset is ~5000 lines and testable term by term; and a 906 KB WASM OpenMM gives us its double-precision answer in Node *and* in the tab |
| What is the engine's input? | `system.xml` + `state.xml` (OpenMM serialisation) + a topology for output | The notebook and the browser then produce the same artefact, so the notebook is the oracle, and anyone's existing OpenMM system runs too |
| Which forces? | `HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`, `NonbondedForce` (PME, exceptions, dispersion correction), `CMMotionRemover`, `MonteCarloBarostat`, the restraint `CustomExternalForce`, and PDBFixer's one soft-core `CustomNonbondedForce` | **Measured**: both the GAFF and the OpenFF/SMIRNOFF route produce exactly these five force types for 3HTB+JZ4 — no virtual sites, no switching function, no other custom force |
| Ligand parameters | **`openff-2.3.0` (Sage 2.3.0) by default**, typed with RDKit WASM SMIRKS matching; GAFF kept as an option but **upload-only** | Verified: `GAFFTemplateGenerator` shells out to `antechamber` for **atom typing**, not just charging, and no GAFF SMIRNOFF port exists — see [§Ligand force field](#ligand-force-field-and-charges) |
| Ligand charges | **OpenFF NAGL exported to ONNX**, run on ONNX Runtime Web; Gasteiger only as a labelled fallback | AM1-BCC needs `sqm`. NAGL is 0.016 e from OpenEye AM1-BCC where AmberTools itself is 0.034 e |
| Force accumulation | **Fixed point**, scattered through emulated 64-bit (carry-propagating) atomics | Fixed-point accumulation is *exact* and order-independent, which is what makes a 1.5-million-kJ/mol cancellation survivable **and** what makes a scattered atomic accumulation bit-reproducible — so gather buys no determinism, only possibly speed. Built and verified in Phase 0 and 1 |
| Positions | **64-bit fixed point with residual feedback** | A 1e-6 constraint tolerance on a 0.1 nm bond is 1e-7 nm; f32 at a 6.5 nm coordinate has a 4.77e-7 nm ulp, so an f32 coordinate does not drift, it stops moving. Double-single was the first answer and is **abandoned**: Chrome/Metal folds Dekker's error term to zero. Measured in [PRECISION.md](PRECISION.md) |
| Integrator | **LFMiddle** | `LangevinIntegrator` *is* `LangevinMiddleIntegrator` in OpenMM ≥ 8.2, so that is what the notebook runs |
| Reproducibility | Every run seeds every RNG and records the seed; a build with the same seed is bit-identical | The notebook's own preparation is not reproducible (measured: 2.3 Å of hydrogen movement between identical runs) |

---

## 1. The contract

The engine never knows how a system was built. Its input is

* `system.xml` — `XmlSerializer.serialize(System)`: particles and masses, periodic
  box vectors, constraints, and the forces above;
* `state.xml` — positions, box vectors, optionally velocities;
* a topology for output (`system.pdb`), which carries the atom names, residues
  and chains that the DCD and the analyses need.

This has four consequences worth stating plainly.

1. **The notebook is the oracle.** `tools/build_reference_system.py` already
   produces exactly these three files from a PDB ID (see
   [measurements/](measurements/)), so every browser-built system can be
   compared with a Python-built one term by term.
2. **Anyone's OpenMM system runs.** Upload `system.xml` + `state.xml` and press
   go. AMBER `prmtop`/`inpcrd` is the second input route.
3. **Stop and resume is free**, which the notebook cannot do: cell 26 rebuilds
   the `System` from live kernel state, so a restarted runtime cannot continue a
   production run even with the `.rst` on disk.
4. **The file sizes are real.** 3HTB at the notebook defaults is 26,078 atoms →
   `system.xml` 7.3 MB, `state.xml` 5.0 MB. Parsing 7 MB of XML on the main
   thread is not acceptable; it happens in a worker, streamed, and the parsed
   form is kept in typed arrays.

## 2. System building

### Route analysis

**Route A — the real Python stack in Pyodide.** Rejected for production.

| Blocker | State |
| --- | --- |
| `rdkit` | no Pyodide wheel; the request is open; upstream calls it a substantial port |
| `openmm` | no emscripten wheels ([#5422](https://github.com/openmm/openmm/issues/5422), open); the C++ core itself *is* solved by Molarium's build |
| `openff-toolkit` | pure Python but its chemistry runs through RDKit, so it inherits that blocker |
| `antechamber`/`sqm` | Fortran/C, never ported — so GAFF and AM1-BCC are unreachable regardless |

Four blockers, not independent: RDKit sits under both `openff-toolkit` and
`prolif`. A realistic estimate is six months or more, almost all of it inside
RDKit's Boost-heavy Python wrapper, for a first-load download in the 150–300 MB
range — and GAFF still would not work. The one thing Route A would buy over
Route B, exact `Modeller`/`PDBFixer` behaviour, is worth less than it sounds,
because that behaviour is **not reproducible even in Python** (§6.1).

**Route B — reimplement the needed subset in TypeScript.** Recommended.

| Piece | What it needs | Rough size |
| --- | --- | --- |
| PDB/mmCIF reader | models, altLoc, insertion codes, element-column fallback, CONECT, SEQRES, LINK | ~500 lines |
| PDBFixer subset | `removeHeterogens`, `findMissingResidues` (SEQRES vs observed), `findMissingAtoms` from the 30 residue templates, `addMissingAtoms` (template superposition + a restrained minimisation with `soft.xml`'s harmonic terms and its one soft-core repulsion `C/((r/0.2)^4+1)`, C = 1) | ~700 lines, and it reuses our own minimiser |
| ffxml parser + residue template matching | `<AtomTypes>`, `<Residues>` with external bonds and variants, `<HarmonicBondForce>`, `<HarmonicAngleForce>`, `<PeriodicTorsionForce>` (proper + improper, wildcard classes), `<NonbondedForce coulomb14scale lj14scale>`; the element+bond-graph match OpenMM uses | ~900 lines |
| `Modeller.addHydrogens` | `hydrogens.xml` definitions, pH-dependent variants, the terminal and His/Cys/Asp/Glu/Lys rules | ~450 lines |
| `Modeller.addSolvent` | tile `tip3p.pdb`/`spce.pdb`, delete clashes with per-atom radii taken from the force field's own sigma, count ions from the ionic strength, replace waters with ions | ~400 lines |
| SMIRNOFF assignment | `.offxml` parser, SMIRKS matching through RDKit WASM, hierarchy (last match wins), charge handlers | ~800 lines |
| `System` XML writer, `prmtop`/`inpcrd` writer | | ~600 lines |

About 4500–6000 lines of TypeScript, 8–12 weeks with the oracle running
alongside. Download cost: **≈1.6 MB** of OpenMM force-field data (ff14SB 224 KB,
amber99sbildn 474 KB, amber99sb 468 KB, the water XMLs, `hydrogens.xml` 13 KB,
and the two 180 KB pre-equilibrated water boxes), plus RDKit MinimalLib
(~1.5 MB wasm), plus the NAGL ONNX model. All cached after first load.

**The hybrid part, and why it matters.** Molarium has already compiled OpenMM
8.2 to a 906 KB WebAssembly module with the double-precision Reference platform
(Apache-2.0 build scripts; OpenMM is MIT). We adopt that build — rebuilt from
its recorded provenance, or rebuilt ourselves from the same patches — for three
jobs, **never in a production path**:

* **`test/*.oracle.test.ts`**: every force kernel and every built system checked
  against real OpenMM double precision in Node, with no Python environment
  needed. This is what makes the acceptance suite runnable by anyone.
* **A "verify this system" button**: the user's own system, single-point energy
  per force group, browser engine against OpenMM Reference, shown as a number
  before they commit to a run. ~1 MB, downloaded only when asked.
* **The same check in CI in a real browser** through Playwright.

`AGENTS.md` will say: *OpenMM-WASM is a test fixture and a verification tool. It
is never a fallback and never runs dynamics for a user.*

### Ligand handling

The ligand comes from the HETATM residues of the selected chains. For 3HTB
chain A that is **four candidates** — `BME`, `PO4` 165, `PO4` 166 and `JZ4` 167
— so the picker lists name, chain and residue number, and waters are dropped
unless chosen.

Bond orders and formal charges come from the RCSB Chemical Component Dictionary
entry, matched onto the PDB coordinates. **Measured on JZ4**: the CCD ideal SDF
gives `CCCc1ccccc1O`, 10 heavy atoms, `AssignBondOrdersFromTemplate` matches all
10, `AddHs` gives 22 atoms at formal charge 0 — the same molecule the notebook
builds from the SMILES `CCCC1=CC=CC=C1O` it asks the user to type. Fallbacks
when the template does not match are in [§Q3](#q3-how-reliable-is-ccd-based-bond-order-assignment).

### Ligand force field and charges

**Default: `openff-2.3.0` (Sage 2.3.0).** Decided; it is also the release NAGL's
authors recommend pairing their charges with, so the default charge method and
the default force field agree rather than being mixed across generations.

`openff-*` and `smirnoff99Frosst-*` run in the browser: SMIRKS matching is what
RDKit WASM already does well. Both the notebook's list and the current releases
are offered — `openmmforcefields` 0.15.1 now ships `openff-2.1.x`, `2.2.x` and
`2.3.0`, none of which existed when the notebook's list was written.

**GAFF stays an option, and in the browser it is upload-only. This was checked
rather than assumed**, three ways:

1. `openmmforcefields.generators.GAFFTemplateGenerator` builds a shell command
   and calls it: `antechamber -i … -fi …` then
   `parmchk2 -i out.mol2 -f mol2 -p gaff.dat …`. Its own comment at the call
   site reads *"Run antechamber without charging (which is done separately)"* —
   so **antechamber is required for atom typing**, and supplying charges does
   not remove the dependency. `sqm` is only the charge half.
2. `openff-forcefields` ships 73 `.offxml` files and **none** of them is GAFF.
   `smirnoff99Frosst` descends from parm99 and parm@Frosst, not from GAFF.
3. OpenFF maintainers treat a complete GAFF→SMIRNOFF port as an open problem,
   and note that the GAFF→Foyer attempt did not reach equivalence.
   `rsdefever/antefoyer` has a SMARTS-based GAFF path, but in Foyer XML, not
   SMIRNOFF, and unvalidated against antechamber.

So browser-native GAFF means reimplementing antechamber's `ATOMTYPE.DEF` rule
engine and validating it against antechamber on a few thousand ligands. That is
its own project with its own parity risk, and it is **not** a Phase 4
deliverable; it is a flagged research task. Until then GAFF works by uploading
parameters made with AmberTools (or a complete `system.xml`/`prmtop`), and the
UI says exactly that where the option appears. Versions offered rise to
`gaff-2.2.20`, which is newer than the notebook's `gaff-2.11`.

**A third option worth offering: `espaloma-0.3.2`**, which `openmmforcefields`
0.15.1 already exposes through `EspalomaTemplateGenerator`. Espaloma is a graph
neural network that predicts the valence parameters *and* the charges from the
molecular graph — no antechamber, no SMARTS, no atom typing at all — so it is
the most browser-native of the three, over the same ONNX path as NAGL. It is a
Phase 4 stretch item, gated against the Python generator like everything else.

Charges are NAGL through ONNX Runtime Web, with the method and model version in
the UI and in `run.json`. Gasteiger exists as a labelled fallback that prints a
warning on every run that uses it.

## 3. Precision

WebGPU has no f64, no 64-bit atomics, and atomics only on `i32`/`u32`. These are
the three decisions that follow, and they go in `AGENTS.md`.

**Why this is not a detail.** Measured on the 3HTB system: PME direct space is
**+1,563,444 kJ/mol** and reciprocal space is **−1,732,547 kJ/mol**, cancelling
to a total potential energy of **−138,029 kJ/mol**. Each term is more than ten
times the answer. A summation scheme that rounds while accumulating loses the
answer in the cancellation, and the loss depends on the order the GPU happened
to run the threads in.

### Force accumulation — fixed point, scattered

Forces accumulate as integers, scaled by 2³² as OpenMM's own GPU platforms do.
Integer addition is exact and commutative, so the sum does not depend on
thread order and nothing is lost while accumulating.

**Corrected during Phase 1.** This section originally said to gather wherever a
gather exists, at 2× the pair work, in order to avoid atomics and so keep the
determinism invariant. That reasoning was wrong, and in a way worth spelling
out: *because* the accumulation is integer, an atomic scatter is already
bit-reproducible. Atomic integer addition is exact and commutative, so the
total does not depend on the order the adds landed in, and the f32 value each
term contributes depends only on that term's own inputs. There is no
floating-point reduction anywhere for an order to matter to. Phase 0 also
measured the atomics as cheap on an Apple M3 Pro — 374 million adds a second
into a *single* contended slot, faster than 65536 uncontended ones, because
scattered memory costs more than contention does.

So:

1. **Scatter through the fixed-point atomics.** One invocation owns one term and
   adds its contribution to the two to four atoms it touches. Built and verified
   for the bonded forces: `src/gpu/kernels/bonded.wgsl`, matching OpenMM to
   between 1.7e-8 and 1.6e-6 relative on energy and our double-precision
   reference to between 5.6e-7 and 7.1e-6 relative RMS on forces, on both
   committed fixtures. Gather stays on the table for the **nonbonded** kernel,
   where it saves a different kind of work, and is a pure performance question
   there.
2. **PME charge spreading** is the densest scatter: each atom spreads to
   5³ = 125 grid points. The carry-propagating atomic path — `atomicAdd` the low
   word, detect the wrap from the returned old value, `atomicAdd` 1 into the
   high word — is **built and verified**, including under heavy contention, and
   is the default. A gather per grid point over a cell-binned atom list remains
   worth measuring purely for speed.
3. **Overflow is checked, not assumed.** The scale is chosen from a measured
   force bound with a recorded headroom factor, and a debug build traps on
   saturation. Pre-minimisation structures are the dangerous case: the
   unminimised 3HTB state has a maximum force component of 5.6e5 kJ/mol/nm.

### Positions — fixed point with residual feedback

**Superseded by measurement during Phase 0; [PRECISION.md](PRECISION.md) is the
current answer and has the numbers.** In summary:

Positions use the same 64-bit fixed-point format as the forces, and the
integrator carries the part of each displacement that did not fit the format
into the next step. Velocities stay f32. All *differences* — bond vectors, pair
vectors, grid offsets — are taken in fixed point and then converted to f32,
where the magnitudes are at most the cutoff and f32's ulp is ~6e-8 nm. An
absolute coordinate is never converted to f32.

The reason positions cannot be f32 is the constraints. The notebook sets
`integrator.setConstraintTolerance(1e-6)`; on a 0.1 nm X–H bond that is 1e-7 nm,
while f32 at a 6.5 nm coordinate has an ulp of 4.77e-7 nm — so a displacement
that small leaves the coordinate completely unchanged. Measured: 200000
displacements of 1e-7 nm move an f32 coordinate not at all, losing 0.02 nm.

The reason they are not double-single, which was the original plan and is the
obvious way to get ~48 bits out of two f32 words: **Chrome/Metal folds Dekker's
error-free arithmetic away.** With runtime inputs, `two_sum`'s error term comes
back as exactly zero, because the backend simplifies `b - (s - a)` into `b - b`.
Nothing warns, and the result is an f32-precision accumulator wearing a
double-single type. Hence the invariant in `AGENTS.md`: no scheme may depend on
an f32 rounding surviving the optimiser.

Residual feedback matters as much as the format. Plain fixed-point accumulation
drifts *linearly in the step count* when a displacement repeats, because the
conversion error is then identical every step rather than random: 2.3e-5 nm over
400 ps, which would be 0.06 nm at a microsecond. Carrying the residual bounds
the outstanding error at one unit in the last place however long the run is —
measured at 1.8e-11 nm over the same 400 ps.

### Determinism

**Invariant: for a fixed seed and neighbour-list rebuild schedule, two runs on
the same device produce bit-identical trajectories.** Nothing accumulates in
floating point; every accumulation is exact integer addition, whose result does
not depend on the order the adds landed in, so atomics are free to race.
Counter-based RNG (Philox 4×32-10) keyed by (step, atom, purpose) means the
random forces do not depend on scheduling either. Note what this does *not*
need: a fixed workgroup configuration. Across devices, bitwise agreement is not
promised, because a different adapter may choose a different kernel variant, and
the reason is recorded.

## 4. The engine

| Component | Decision |
| --- | --- |
| Bonded | `HarmonicBondForce`, `HarmonicAngleForce`, `PeriodicTorsionForce`; per-atom term lists for gather |
| Direct space | LJ + real-space Ewald, cutoff 1.0 nm, no switching function (measured: `useSwitchingFunction = false`), exceptions with the per-pair charge/sigma/epsilon the `System` carries — AMBER 1-4 scaling for the protein and the ligand force field's own scaling, both read from the file rather than assumed |
| Neighbour list | cell list + Verlet list with skin, rebuilt on a displacement criterion; full list for gather; periodic |
| Dispersion correction | analytic long-range correction, on by default (measured: `useDispersionCorrection = true`) |
| PME | α = `sqrt(-log(2·tol))/cutoff` = **2.628261 nm⁻¹** for the notebook's settings; minimum grid `ceil(2·α·L/(3·tol^0.2))`, so 53 for 3HTB's 6.5239 nm box and 25 for 1L2Y's 3.1127 nm box, rounded up to the nearest 2/3/5/7-smooth size; order-5 B-splines; a WGSL 3D FFT as three batched 1D Stockham passes. Both the minimum and the rounded grid go in `run.json`; a run can pin an explicit grid to compare with a given OpenMM platform |
| Constraints | SETTLE for rigid water, CCMA for bonds to hydrogen, tolerance 1e-6. 3HTB has 24,719 constraints out of 26,078 particles — almost every atom is in one, so this is a hot kernel, not a detail |
| Integrator | LFMiddle (= OpenMM ≥ 8.2 `LangevinIntegrator`), friction 1 ps⁻¹; optional hydrogen mass repartitioning, off by default |
| Barostat | `MonteCarloBarostat`, isotropic, **default frequency 25 steps**, molecule-wise centroid scaling, Metropolis on ΔE + PΔV − NkT·ln(V′/V) |
| Restraints | `k·periodicdistance(x,y,z,x₀,y₀,z₀)²` with per-particle k, x₀, y₀, z₀ — that expression, hard-coded, not a general parser. Same for PDBFixer's `C/((r/0.2)^4+1)` |
| Minimiser | L-BFGS on the GPU, as OpenMM's `LocalEnergyMinimizer` does, stopping on tolerance (10 kJ/mol/nm) or `maxIterations`. Anchor: 10,000 iterations on the 26k-atom system took **57.9 s** on this machine's CPU platform |
| Run loop | steps submitted in chunks sized to stay under the GPU watchdog; readback only at reporting intervals; checkpoint to OPFS every stride; Screen Wake Lock; a warning that a background tab may be throttled; the stride model kept so a long run can stop and resume |
| `device.lost` | stop, keep the last checkpoint, offer resume |

## 5. Repository layout

```
src/engine/      System/State parsing, run loop, reporters, checkpoints
src/gpu/         device + limits negotiation, allocator, pipeline cache
src/gpu/kernels/ WGSL
src/prep/        PDB/CCD reading, PDBFixer subset, ffxml, Modeller, SMIRNOFF
src/analysis/    RMSD/RMSF/Rg/2D-RMSD/PCA/correlation, interactions, writers
src/reference/   CPU double-precision TypeScript reference implementation
web/             UI
tools/           Python oracle generators and probes, benchmarks
test/            Vitest, *.gpu.test.ts (Dawn), *.oracle.test.ts (OpenMM-WASM)
test/browser/    Playwright
reference/       the notebook
docs/            PLAN, NOTEBOOK_PARITY, PRIOR_ART, DEVIATIONS, measurements
```

`AGENTS.md` holds the invariants: units are OpenMM's (nm, ps, kJ/mol, elementary
charge); atom ordering is the `System`'s; no silent physics changes; every GPU
kernel has a deterministic differential test against an independent reference;
reference data and tolerances are never adjusted to make a test pass;
OpenMM-WASM is a fixture, not a fallback. `CLAUDE.md` holds the operational
half: how to install, build, test, benchmark and regenerate fixtures in this
checkout. `THIRD_PARTY.md` lists every bundled file with its license. App code
is MIT.

## 6. Validation

Three questions that cannot see each other's failures, so three lanes.

### 6.1 Does the browser build the same system?

The honest constraint first: **the notebook's own preparation is not
reproducible.** `Modeller.addHydrogens` places any hydrogen it cannot derive
geometrically at a random position from Python's unseeded global RNG and then
relaxes it with 50 minimiser iterations at a 1.0 kJ/mol/nm tolerance;
`addSolvent` picks which water becomes an ion with `random.choice`. Measured,
three identical preparations of 3HTB chain A:

| | trial 0 | trial 1 | trial 2 |
| --- | --- | --- | --- |
| atoms | 2614 | 2614 | 2614 |
| bounding radius (nm) | 3.084629 | 3.082956 | 3.076976 |
| box at 4 Å padding (nm) | 6.569259 | 6.565913 | 6.553951 |
| max per-atom displacement vs trial 0 | — | **2.35 Å** | **2.33 Å** |

So "same number of waters" is not a criterion anyone can meet, and the water
count also moves with the *ligand* force field, because `addSolvent` takes its
clash radii from the force field's own sigma values (7798 waters with gaff-2.11,
7953 with openff-2.0.0 for the same solute). The parity check is therefore
split:

* **Parameters — exact.** Same atoms and order for the solute (or a documented
  mapping), and every force term's parameters equal to the Python build's:
  bond and angle k and r₀/θ₀, torsion periodicity/phase/k, per-particle charge,
  sigma and epsilon, every exception, every constraint. This is a term-by-term
  comparison of two `system.xml` files and it must match exactly, bar
  floating-point formatting.
* **Energies — cross-fed.** Our system evaluated at *their* coordinates and
  theirs at ours, per force group, from the double-precision reference. This
  separates a parameter error from a geometry difference, which comparing two
  independently built systems cannot do.
* **Charges — reported, not matched.** NAGL against AmberTools AM1-BCC, per
  atom, with the RMSE and the worst atom in the report and in the UI.
* **Solvation — statistical.** Water count within a stated band, density at
  298 K/1 bar, the minimum solute-to-water distance, the O–O radial
  distribution against a TIP3P reference, and the ion count against the
  ionic-strength formula.
* **Reproducibility — better than the notebook.** Seed everything, record the
  seed, and gate that the same seed gives a byte-identical `system.xml`.

### 6.2 Do the kernels compute what OpenMM computes?

Per-force-group energies and per-atom forces against OpenMM's `Reference`
platform for the same `system.xml` and `state.xml`, in two lanes: the Python
oracle (`tools/`) and OpenMM-WASM in Node and in the browser. The first
committed fixture, from `measurements/3HTB-JZ4-facts.json`:

| force group | energy (kJ/mol) |
| --- | ---: |
| `HarmonicBondForce` | 1,194.202 |
| `HarmonicAngleForce` | 15,306.220 |
| `PeriodicTorsionForce` | 14,574.340 |
| `NonbondedForce` (direct) | 1,563,444.215 |
| `NonbondedForce` (reciprocal) | −1,732,547.546 |
| `CMMotionRemover` | 0.000 |
| total | −138,028.569 |

Acceptance: WebGPU against Reference no worse than OpenMM `CUDA` mixed
precision against Reference, within a small factor. Those thresholds are
**calibrated, not guessed** — `tools/calibrate_thresholds.py` minimises and runs
20 ps of NVT first (measured at the fixture's raw starting coordinates, the
relative force error says more about the input structure than about the
platform), then compares every platform and precision this host offers against
the double-precision Reference.

Provisional numbers, equilibrated 1L2Y (`measurements/calibration-M3Pro-opencl.json`):

| basis | force rel. RMS vs Reference | total energy error | NVE drift at 2 fs |
| --- | ---: | ---: | ---: |
| OpenCL single, Apple M3 Pro | 1.05e-05 | −0.0019 kJ/mol | −0.168 kJ/mol/ns/atom |
| CPU, double | 8.6e-07 | −0.0046 kJ/mol | −0.123 kJ/mol/ns/atom |

Two things worth reading off that table. **A legitimate single-precision
implementation lands near 1e-5 relative force error** — that is the floor, not
1e-7, and it is what our engine should be measured against. And **NVE drift is
mostly algorithmic, not precision**: single precision adds about 40% over
double, so a drift far above ~0.17 kJ/mol/ns/atom means a bug, not rounding.

These are provisional because the basis is single, not mixed, precision. The
authoritative set comes from a **Colab T4** run of the same script — Pablo's
machine for it — since this Mac's OpenMM has no mixed-precision GPU platform
(`OpenCL/mixed` reports "No compatible OpenCL platform is available").

A TypeScript double-precision reference implementation in `src/reference/` sits
between the two: it is the independent implementation each WGSL kernel is
differentially tested against, and it runs in plain Vitest with no GPU.

### 6.3 Does the dynamics behave?

* NVE energy drift with constraints at 2 fs, against OpenMM mixed precision.
* NVT temperature distribution against the Maxwell–Boltzmann prediction.
* NPT density of a pure TIP3P box at 298 K and 1 bar.
* Constraint satisfaction below the tolerance for the whole run.
* A short protein–ligand run whose RMSD, RMSF, Rg and ligand-RMSD
  distributions agree statistically with an OpenMM run of the same length.
* Each analysis gated against its notebook counterpart (pytraj, MDAnalysis,
  ProLIF) on a committed trajectory.

### 6.4 Test lanes

| Lane | Command | What it needs |
| --- | --- | --- |
| Unit + TS reference | `npm test` | nothing |
| GPU differential | `npm run test:gpu` | Dawn through npm `webgpu`; `tools/gpu-chrome.mjs` as the fallback lane when the binding will not load on a given glibc |
| OpenMM oracle | `npm run test:oracle` | the OpenMM-WASM module |
| Browser end to end | `npm run test:browser` | Playwright + Chromium; includes one full PDB ID → analysis run on 1L2Y |
| Benchmark | `npm run bench` | reports ns/day for DHFR and the test systems with the adapter name |
| Fixtures | `python tools/build_reference_system.py …` | the conda oracle environment |

No large binary fixtures in Git. `facts.json` and the acceptance numbers are
committed; `system.xml`, `state.xml` and trajectories are regenerated or
attached to a release.

## 7. Phases

Each phase ends with its tests passing, benchmark numbers, an updated
`HANDOFF.md`, and a short summary.

**Phase 0 — skeleton (days).** Vite + TypeScript strict, `AGENTS.md`,
`CLAUDE.md`, `THIRD_PARTY.md`, the test lanes wired up empty, the oracle
environment pinned in `tools/environment.yml`, GitHub Pages workflow.
*Gate:* `npm run check` passes on an empty tree; `tools/` regenerates the two
committed fixtures.

**Phase 1 — engine core (the long one).** `System`/`State` parser; the TS
double-precision reference; WGSL bonded, direct-space, PME, constraints,
LFMiddle, barostat, restraints, minimiser. Built against the fixtures in this
order: bonded → direct space with exceptions → PME reciprocal → constraints →
integrator → barostat.
*Gates:* per-force-group energies and per-atom forces inside the calibrated
thresholds for 1L2Y, 3HTB and DHFR; NVE drift at 2 fs; a measured ns/day for
each; the determinism invariant gated by a repeat-run test.

**Phase 2 — first release: run an OpenMM system in the browser.** Upload
`system.xml` + `state.xml` + PDB, or `prmtop`/`inpcrd` → minimise → equilibrate
→ production by strides → DCD, logs, plots, zip. GPU panel, live plots, ns/day
and remaining time, OPFS checkpoints, wake lock, `run.json`, the "verify this
system" button. Deployed to GitHub Pages, marked experimental.
*Gates:* a 1 ns 1L2Y run end to end in a browser under Playwright; DCDs open in
MDAnalysis, MDTraj, pytraj and VMD; resume after a deliberate `device.lost`.

**Phase 3 — protein in water from a PDB ID.** The PDBFixer subset, ffxml and
template matching, `addHydrogens`, `addSolvent`, ff14SB/ff99SB/ff99SB-ILDN,
TIP3P/SPC/E, padding, ions, pH; `prmtop`/`inpcrd` export; the protein analyses
(Cα RMSD, Rg, RMSF, 2D RMSD, PCA, cross-correlation) with their distributions,
PNG and CSV.
*Gates:* term-by-term parameter parity against Python for 1UBQ, 1L2Y and DHFR;
cross-fed energies; the solvation statistics; each analysis against pytraj on a
committed trajectory.

**Phase 4 — protein–ligand from a PDB ID.** Heterogen picker, CCD bond orders,
SMIRNOFF typing with RDKit WASM, NAGL in ONNX, GAFF by upload, and the ligand
analyses (~~interaction network~~, ligand RMSD, pocket and named-residue
distances, frame extraction).

🔴 **The interaction network is cut**, by decision rather than by difficulty.
Reproducing ProLIF's fingerprint means reproducing nine interaction types whose
criteria rest on atom typing a PDB does not carry, and the parts that were
built — chemistry perception and the hydrophobic rule — agreed with ProLIF
exactly while the assembled per-residue fingerprint did not (21 of 25 pairs
differing in at least one frame). Shipping that behind a loose tolerance would
have been worse than not shipping it, and the notebook's own version is already
coarse: deviation 11 records that it samples at most ten frames whatever the
trajectory length. Anyone wanting it has ProLIF.
*Gates:* parameter parity for 3HTB/JZ4 and 181L/benzene against the OpenFF
Python route; the NAGL-vs-AM1-BCC charge report; the interaction network against
ProLIF.

**Phase 5 — polish.** Subgroups, workgroup tuning, fewer readbacks, Safari and
Apple-GPU and integrated-GPU testing, accessibility, GROMACS export best
effort, documentation and a tutorial page mirroring the Colab notebook.

## 8. Risks

| Risk | Mitigation |
| --- | --- |
| **PME in WGSL is the single biggest technical risk** — no TS/WGSL 3D FFT exists, and the direct/reciprocal cancellation is unforgiving | Build it first, against the reciprocal-space energy of a committed fixture; keep an Ewald-sum (no FFT) path as the differential reference for small systems, where the exact answer is computable |
| NAGL does not export cleanly to ONNX (DGL message passing, dynamic graph size) | Molarium's ANI-2x → ONNX is precedent. If it fails: a Colab helper notebook produces the charge set and the user pastes or uploads it; the browser path stays usable |
| Residue template matching has a long tail (nonstandard residues, disulfides, His protonation variants, chain breaks, D-amino acids) | The oracle tells us immediately; the app refuses with a clear message rather than guessing, and the refusal names the residue |
| GPU watchdog resets and background-tab throttling on long runs | Chunked submission sized by measurement, OPFS checkpoints per stride, wake lock, and an explicit warning |
| Dawn's Node binding not loading on a given platform | Keep `tools/gpu-chrome.mjs` (LocalFold's lane) as the second GPU lane from Phase 1, not as a later rescue |
| Safari and Apple GPU differences | Phase 2 deploys and tests on this M3 Pro; Safari is a Phase 5 gate, not an assumption |
| Scope | The phase boundaries are release boundaries. Phase 2 is useful on its own to anyone who already has an OpenMM system |

---

## Answers to the six open questions

### Q1 — Route A, Route B, or a hybrid?

**A hybrid, weighted to B: TypeScript for production, real OpenMM in
WebAssembly as the oracle and as a user-facing verification button.**

* *Effort.* Route B's preparation layer is 4500–6000 lines of TypeScript,
  8–12 weeks with the oracle alongside. Route A is six months or more, almost
  all of it porting RDKit to Pyodide, and it still cannot do GAFF because `sqm`
  has never been ported.
* *Download size.* Route B: ~1.6 MB of force-field data, ~1.5 MB RDKit
  MinimalLib, the NAGL ONNX model, ~1 MB of OpenMM-WASM only when verification
  is requested. Route A: 150–300 MB, every session that clears its cache.
* *Fidelity.* Route B can be **exact** where exactness is meaningful — every
  force term's parameters, verified term by term — and cannot be exact on
  solvation geometry. Neither can Python: three identical preparations of 3HTB
  move hydrogens by 2.3 Å and the box by 15 pm (§6.1). We therefore make the
  browser *more* reproducible than the notebook by seeding and recording, and
  we validate solvation statistically.

The hybrid's real win is testing: OpenMM's own double-precision answer becomes
available in Node and in a tab, so the acceptance suite needs no conda
environment and the user can check their own system.

### Q2 — Ligand charges in the browser

**OpenFF NAGL, exported to ONNX, run on ONNX Runtime Web.** Code MIT, model
`openff-gnn-am1bcc-1.0.0.pt` (AshGC 1.0) CC BY 4.0, so it can ship with
attribution.

*Expected difference from the notebook.* NAGL predicts **OpenEye AM1-BCC
ELF10** at a reported 0.016 e atomic-charge RMSE over 52,313 molecules. The
notebook does not use OpenEye — `openmmforcefields` charges through AmberTools
`sqm`, which itself differs from OpenEye ELF10 by 0.034 e. So the
browser-versus-notebook charge difference should be of order **0.03 e RMSE**,
dominated by the backend disagreement that already exists inside the Python
ecosystem, not by NAGL. Larger deviations are reported for small molecules,
anions, and sulfur/phosphorus chemistry, and NAGL's authors recommend pairing it
with Sage 2.3.0 rather than swapping charge backends under an older force field —
which is a reason to offer current Sage, not only the notebook's list.

Sage 2.3.0 being the chosen default is what makes this coherent: NAGL's authors
recommend pairing AshGC with Sage 2.3.0 rather than swapping charge backends
under an older force field, so the default force field and the default charge
method now come from the same generation.

This is measured per test ligand before NAGL becomes a default, published in the
report, and stated in the UI and `run.json`. Alternatives, in order: an uploaded
SDF carrying charges; uploaded AmberTools parameters; a charge set computed in a
Colab helper notebook and pasted in; Gasteiger, labelled unsuitable for
production.

### Q3 — How reliable is CCD-based bond-order assignment?

**For a ligand that came out of a PDB entry, coverage is essentially complete by
construction** — every HETATM residue in the PDB has a Chemical Component
Dictionary entry, and the ideal SDF carries bond orders, formal charges and
coordinates. Verified live: `JZ4.cif` and `ATP_ideal.sdf` both fetch, and JZ4
assigns in one call with all 10 heavy atoms matched and the right SMILES.

What actually fails, in rough order of frequency:

1. **Atoms missing from the deposited model** (disorder, partial occupancy). The
   template has more atoms than the fragment and the match fails.
2. **Alternate locations** — 3HTB chain A has 64 atoms in two conformations.
3. **Covalent ligands** (a LINK record to a protein atom). The ligand is not a
   separate molecule and the whole one-molecule assumption breaks.
4. **Branched or polymeric components** — oligosaccharides split across several
   CCD entries joined by LINK records.
5. **Metals, clusters and ions**, which need a different treatment entirely.
6. **Protonation.** The CCD describes a reference state, not necessarily the
   bound state at pH 7.4 — carboxylates, amines, phosphates. This is the one
   that quietly changes the physics, and no automatic rule fixes it.

The fallback flow, in the UI, in this order: show the perceived molecule as a 2D
depiction plus SMILES plus formal charge and ask the user to confirm; on
mismatch, attempt a maximum-common-substructure match against the template and
show exactly which atoms are missing; offer "upload an SDF" and "paste a
SMILES"; for cases 3–5, refuse with the reason named. Nothing proceeds silently,
and `run.json` records which route was taken.

### Q4 — Force accumulation and position precision

See §3 and [PRECISION.md](PRECISION.md) for the measured answer, which differs
from this plan's first one in two places. In short: **fixed-point accumulation
scattered through emulated 64-bit atomics, fixed-point positions with residual
feedback, counter-based RNG**, with the invariant that a seed reproduces a
trajectory bit-for-bit on one device. Gather is not needed for determinism —
integer accumulation is order-independent — and double-single positions are
unusable because Chrome/Metal folds the error-free arithmetic they rest on.

*Accuracy cost, now partly measured.* The bonded forces agree with our
double-precision reference to between 5.6e-7 and 7.1e-6 relative RMS, and their
energies agree with OpenMM's Reference platform to between 1.7e-8 and 1.6e-6
relative, on both fixtures. PME is expected around 1e-5 relative, limited by f32
B-spline and FFT arithmetic rather than by accumulation, which is exact. Energy
conservation should match OpenMM mixed precision, because the scheme is the same
class. The remaining cost is performance rather than accuracy, and it is smaller
than this plan first assumed: no full neighbour list is needed for determinism's
sake.

### Q5 — Largest system and expected ns/day

*Memory.* About **1.1 kB per atom**, dominated by the neighbour list: at a 1.0 nm
cutoff with a 0.2 nm skin, a water-density sphere holds ~240 neighbours, so a
full list is ~1 kB/atom, against ~75 B/atom for positions (double-single),
velocities and fixed-point forces. The PME grid is negligible (64³ complex f32
is 2 MB). So 26k atoms is ~29 MB and 250k atoms is ~280 MB.

*The binding limit, not the memory, is the first wall.* WebGPU gives a device a
128 MiB storage-buffer binding and a 256 MiB maximum buffer size unless it asks
for more. At 1 kB/atom one bound neighbour list reaches 128 MiB at ~128k atoms,
so the engine either raises the limits (most desktop adapters grant up to
~2 GiB) or binds the list in windows, as AlphaFold2 WebGPU does for its pair
tensor. The GPU panel computes and shows this ceiling before a run starts.

*Practical ceilings*, to be replaced by measurement: ~250k–500k atoms on an
8–24 GB discrete GPU, ~100k–200k on an M-series with unified memory,
~30k–60k on a typical integrated laptop GPU.

*ns/day.* There is now a **measured** native-GPU anchor, not just a projection.
OpenMM's OpenCL platform does work on this machine in single precision (its
`mixed` mode does not), so the 26,078-atom 3HTB system at 2 fs with PME runs at:

| platform on this host | ns/day, 26,078 atoms | source |
| --- | ---: | --- |
| **Apple M3 Pro GPU**, OpenMM OpenCL single | **112.8** | `measurements/calibration-M3Pro-opencl.json` |
| Apple M3 Pro CPU (12 cores), OpenMM CPU | 16.0 | same |

DHFR is 23.5k atoms, so those numbers carry over almost directly. OpenCL on
Apple is a deprecated backend that nobody tunes, so 112.8 ns/day is a floor for
what the hardware can do, not a ceiling.

From that anchor, WebGPU projections at 1.5–3× off native for bandwidth-bound
kernels, plus the barostat readbacks and the browser's submission granularity:

| | projected ns/day for DHFR |
| --- | --- |
| Recent discrete NVIDIA | 100–400 |
| **Apple M-series (M3 Pro class)** | **35–75** |
| Typical integrated laptop GPU | 5–20 |

The NVIDIA row is still a projection off OpenMM's published CUDA figures and
gets replaced by the Colab T4 calibration run plus `npm run bench` at the end of
Phase 1. None of this is a cross-device claim.

### Q6 — What should not be ported

Already settled by instruction: the entire docking half (P2Rank, grid box,
SMILES/MOL ligand input, TorchANI, Uni-Dock, pose selection) — see the table at
the top of [NOTEBOOK_PARITY.md](NOTEBOOK_PARITY.md).

Beyond that:

1. **GNINA rescoring (cells 31–32) and the score print in cell 33.** A CUDA
   binary with no browser route. The *frame extraction* in cell 33 is kept.
2. **`trr`/`xtc` trajectory output.** GROMACS formats that nothing in this
   pipeline reads; `dcd` and `pdb` cover the tools the notebook itself uses.
   Revisit only if asked.
3. **GROMACS `top`/`gro` export — best effort, Phase 5, and labelled.** It comes
   from ParmEd, whose GROMACS writer has its own conventions, and the notebook
   has to build a *second*, constraint-free `System` to produce it at all.
   Exact ParmEd parity is a poor use of effort against a format no step of this
   protocol consumes. AMBER `prmtop`/`inpcrd` is kept, because the analysis path
   needs it.
4. **`pt.bfactors(byres=True)` in the RMSF cell.** Computed and never used.
5. **The Google Drive mount, the conda/pip installation cells, and
   `nvidia-smi`.** No environment to install; the adapter panel replaces the GPU
   check.
6. **`show_box` as a solvent-accessible surface.** It is mislabelled; the app
   draws the real periodic box and offers the surface separately.
7. **Retyping run parameters into the analysis cells.** Everything comes from
   `run.json`, so a plot's time axis cannot disagree with its trajectory.
8. **The pH field in the topology step as a second, independent control.** It
   has no effect on this path (measured); there is one pH, applied where
   protonation is actually decided.

One thing the notebook does *not* do that the app must: warn about periodicity
from the geometry rather than from the padding value. **Measured**: 1L2Y at the
notebook defaults gives a 3.1127 nm box in which the nearest solute-to-own-image
distance is **0.958 nm — inside the 1.0 nm cutoff.** 3HTB gives 1.822 nm and is
fine. The check is cheap (a KD-tree over the solute after solvation), so the app
computes the real distance, shows it, and warns when it is below the cutoff. The
`padding < cutoff` heuristic stays as the pre-solvation hint, with the box
semantics stated correctly: OpenMM's padding is the **total** added width,
`max(2·radius + padding, 2·padding)` around the bounding **sphere**, so the
notebook's "(solute size) + 2×(padding)" and "no closer than 1 nm to any edge"
are both wrong.

---

## Decisions taken

Settled by Pablo on 2026-10-19; `docs/DEVIATIONS.md` carries the consequences.

| | Decision |
| --- | --- |
| Restraint mask | **Ship both**, defaulting to heavy atoms (`!(@H*) & …`), with the notebook's own mask available and recorded in `run.json` |
| Ligand force field | Default **`openff-2.3.0`**. GAFF kept as an option; verified that it needs `antechamber` for atom typing, so in the browser it is upload-only, and browser-native GAFF typing is a flagged research task rather than a Phase 4 deliverable |
| Molarium | No contact needed. The Apache-2.0 build scripts are used under their license, with attribution in `THIRD_PARTY.md` |
| Threshold calibration | **Colab T4**, via `tools/calibrate_thresholds.py`. The committed M3 Pro numbers are provisional until that run replaces them |
| Phase order | Confirmed. Phase 2 — upload an OpenMM system and run it — is the first release |

## Next

Phase 0 and Phase 1, in that order, unless told otherwise:

1. Phase 0 skeleton — Vite, TypeScript strict, `AGENTS.md` with the calibrated
   thresholds, `CLAUDE.md`, `THIRD_PARTY.md`, the five test lanes wired up
   empty, the Pages workflow.
2. Phase 1 engine core, built against the committed fixtures in the order
   bonded → direct space → PME → constraints → integrator → barostat, with the
   TypeScript double-precision reference written first so each WGSL kernel has
   something independent to be differentially tested against.

The one thing worth doing before Phase 1: run `tools/calibrate_thresholds.py`
on the Colab T4 for 1L2Y, 3HTB and DHFR, so Phase 1's gates are real numbers
from the start rather than placeholders.
