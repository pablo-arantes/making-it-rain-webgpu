# Prior art

What already exists for running this stack in a browser, what state it is in,
and what its license permits. Checked 2026-10-19. Repository claims were read
from the sources themselves (cloned and inspected) unless marked otherwise.

**The short version.** Somebody has already compiled OpenMM to WebAssembly and
already assigns SMIRNOFF parameters in a browser with RDKit — both in one
Apache-2.0 project, Molarium. Nobody has run **explicit solvent with periodic
boundaries and PME** in a browser. That is the part this project has to build,
and it is also the part the notebook's protocol is entirely made of.

---

## 1. The nearest neighbour: Molarium

[`rafwiewiora/molarium`](https://github.com/rafwiewiora/molarium) — "a
local-first molecular viewer, builder, and simulation workbench", Apache-2.0,
original code © 2026 Rafal P. Wiewiora. Preprint September 2026.

| Piece | State | Use to us |
| --- | --- | --- |
| **OpenMM 8.2.0 compiled to WebAssembly** — `openmm/build-wasm.sh`, two patches, a 578-line C bridge, `BUILD-PROVENANCE.json` pinning source SHA-256, Emscripten 6.0.6-git, CMake 3.31.0, and output hashes. Double-precision **Reference platform only**; CPU/CUDA/OpenCL/HIP plugins off; WASM exceptions, growing memory, **no pthreads**, no virtual filesystem. The shipped `molarium-openmm.wasm` is **906 KB**. | Working, with a recorded native-parity report | **Large.** It proves the core compiles, and it makes OpenMM's own double-precision answer available *inside the browser and inside Node* with no Python. This is the acceptance oracle the engine needs, available in CI on any machine. OpenMM itself is MIT; the build scripts are Apache-2.0. |
| A browser-only patch making Reference-platform CCMA setup serial, because unpatched OpenMM "tries to create a Reference-platform thread pool while constructing a constrained `Context` and deadlocks in browser WebAssembly" | Working | A trap we would otherwise have hit on our first constrained system |
| **`openff/sage-parameterizer.js`** — 349 lines assigning OpenFF Sage 2.1.0 bonds, angles, proper and improper torsions, vdW and constraints in the browser, by matching each parameter's SMIRKS with RDKit WASM and keeping the last match per atom tuple | Working; validated 22/22 cases against native OpenMM on M1 Pro and L4 | **Large.** This is the Route B ligand path, demonstrated. Our version needs the charge handler, which theirs does not have (see below). |
| `rdkit/` — a patched RDKit 2025.03.4 MinimalLib exposing MMFF94/UFF energies and BFGS, ETKDGv3, Gasteiger charges, Dimorphite-DL protonation, and **mapped SMIRKS matches** | Working | The patch exists because stock MinimalLib returns match atoms in query order without the SMIRKS map numbers. We can either patch as they did or parse the map numbers out of the SMIRKS ourselves (see PLAN) |
| `webgpu/molarium-webgpu.wgsl` — 1048 lines: bonded terms, full nonbonded with **no cutoff**, SHAKE + RATTLE, OBC2/ACE implicit solvent, f32 | Working, user-facing | **Reference only.** No periodic boundaries, no PME, no barostat, no neighbour list. Their README says so plainly. Useful for SHAKE/RATTLE and for the shape of a WGSL force kernel. |
| **ANI-2x exported from TorchANI 2.8.1 to ONNX**, run with ONNX Runtime Web 1.27.0 on WebGPU | Working, ≤ 96 atoms | **Precedent for the NAGL plan**: a PyTorch chemistry model, exported to ONNX, running on WebGPU in a tab |
| Charges | Gasteiger in the browser, described by its own README as "a browser MVP, not Sage's preferred AM1-BCC or NAGL workflow"; the prepared protein fixtures carry **NAGL charges computed ahead of time, server-side** | The open problem is the same one we have, and they did not solve it in the browser |
| Protein support | "experimental Sage/Gasteiger parameters for arbitrary proteins" plus exported OpenMM systems for prepared cases (ubiquitin, Trp-cage); no AMBER protein force field, no loops, no membranes, no periodic solvent | Our P3 (ff14SB + TIP3P + ions from a PDB ID) is not covered |

**Conclusion.** Molarium and this project overlap on infrastructure and barely
overlap on purpose: it is a modelling and docking workbench doing non-periodic
implicit-solvent calculations; this is a port of one specific explicit-solvent
NPT protocol. Worth writing to Rafal Wiewiora before Phase 1 — the OpenMM-WASM
build is the single biggest piece of work either project would duplicate, and
the patches belong upstream in OpenMM issue #5422 either way.

## 2. The engineering pattern: AlphaFold2 WebGPU and LocalFold

| Repository | License | What to take |
| --- | --- | --- |
| [`martin-steinegger/alphafold2-webgpu`](https://github.com/martin-steinegger/alphafold2-webgpu) | **no LICENSE file and no `license` field in `package.json`** — all rights reserved. Study only; copy nothing | The whole validation culture: a nine-line `AGENTS.md` of invariants ("never change reference tensors or tolerances merely to make a failing test pass", "every GPU kernel must have a deterministic differential test against an independent reference implementation"); `tools/*.py` capturing official intermediates as fixtures; `test/*.gpu.test.ts` run in Node through Dawn (`npm i webgpu`); `test/browser/*.spec.ts` in Playwright; per-device kernel *calibration* (compile every variant, check against a reference, then time it, and only then choose); device limits negotiated per problem shape with a pre-flight memory estimate instead of a fixed limit; large weights shipped as GitHub **release assets** that the Pages workflow downloads, verifies by SHA-256 and byte length, and then deploys |
| [`sokrypton/localfold`](https://github.com/sokrypton/localfold) | **Beer-ware (Revision 42)** — permissive in effect; attribute anyway | A no-build-step `src/` loadable straight by the browser; `CLAUDE.md` as the operational half of `AGENTS.md` with ~500 measured findings indexed into `docs/`; `HANDOFF.md` holding only what is *unfinished*; `docs/PARITY.md` separating "our kernels compute the reference's kernels" from "do the two produce the same answer"; `tools/gpu-chrome.mjs` driving headless Chrome when the Dawn Node binding will not load; `oracle-dumps/`; and `src/chem/` — SMILES, kekulisation, rings, stereo and conformers in plain JS, gated against RDKit at 74/74 graph, 43/43 stereo, 1148/1622 rewrite cases. That last one is the proof that browser-side chemistry can be held to a reference instead of hoped about |

Neither runs MD, and neither has a force field. The transferable asset is the
method: nothing is called done until it is differentially tested against the
official implementation, and the thresholds are never relaxed to make a test
pass.

## 3. Can the real Python stack run in the browser?

| Component | Pyodide / browser status | Note |
| --- | --- | --- |
| Pyodide (current package list, 314.0.7) | `numpy` 2.4.6, `scipy` 1.18.0, `networkx` 3.6.1, `pydantic` 2.12.5, `lxml` 6.1.3 are present | the scaffolding is there |
| `rdkit` (Python) | **not in Pyodide.** The wheel request ([rdkit-pypi#81](https://github.com/kuelumbus/rdkit-pypi/issues/81)) is open; upstream describes a substantial port. The documented browser route is RDKit.js called from Python through the JS bridge, which is not `from rdkit import Chem` | blocker |
| `openmm` | **not in Pyodide.** [openmm#5422 "Providing emscripten wheels"](https://github.com/openmm/openmm/issues/5422), opened 2026-09-10, **open**. Peter Eastman: *"Have you tried compiling for emscripten? Does it require changes to the code or build scripts? I assume you only include the reference and CPU platforms?"* A contributor pointed at Molarium's build as a working example. No maintainer commitment, no wheels | blocker for the Python route; the C++ core itself is **solved** by Molarium |
| `pdbfixer` | not packaged, but pure Python over `openmm` + `numpy` — it imported and ran here straight from a `git clone` with nothing installed | easy *if* `openmm` existed |
| `openff-toolkit` | **not in Pyodide.** Pure Python, but its chemistry goes through RDKit or OpenEye, so it inherits the RDKit blocker. Conda-forge is the only supported install | blocker |
| `openmmforcefields` | not packaged; needs `openmm`, `openff-toolkit`, and for GAFF an `antechamber` binary | blocker |
| `mdtraj`, `MDAnalysis`, `openbabel`, `pytraj`/`cpptraj`, `prolif` | none in Pyodide; all have compiled cores (except ProLIF, which needs RDKit and MDAnalysis) | analysis must be reimplemented |
| AmberTools `sqm` / `antechamber` | Fortran/C, not ported, not attempted anywhere found | this is why AM1-BCC cannot run in a tab |

So the Python route is blocked in four places at once, and the blockers are not
independent — `rdkit` is underneath both `openff-toolkit` and `prolif`.

## 4. Chemistry in the browser

| Component | Version / license | What it gives |
| --- | --- | --- |
| **RDKit.js** `@rdkit/rdkit` | 2026.9.1, **BSD-3-Clause**, "RDKit MinimalLib compiled to WebAssembly" | `get_mol`, `get_qmol` (SMARTS/SMIRKS), `get_substruct_matches` with `{useChirality, uniquify, maxMatches}`, Gasteiger charges, ETKDGv3 embedding, MMFF94/UFF. Enough for SMIRNOFF typing, CCD template matching and ligand perception. Matches come back as target atom indices **in query-atom order**, so SMIRKS map numbers have to be recovered from the pattern (or exposed by a MinimalLib patch, as Molarium did) |
| **RCSB Chemical Component Dictionary** | per-component files, free to use | `https://files.rcsb.org/ligands/download/{CODE}_ideal.sdf` (bond orders, formal charges, ideal coordinates) and `{CODE}.cif`. Both verified live: `JZ4.cif` 5963 B, `ATP_ideal.sdf` 3312 B. This is what replaces "the user types a SMILES" |
| **OpenFF NAGL** | code **MIT**, released models **CC BY 4.0**; `openff-gnn-am1bcc-1.0.0.pt` (AshGC 1.0) | a graph neural network predicting AM1-BCC charges. Reported 0.016 e atomic-charge RMSE against OpenEye AM1-BCC ELF10 over 52,313 molecules, where AmberTools' own AM1-BCC backend is 0.034 e against the same reference.¹ Ships as PyTorch/DGL; **no official ONNX export** — that is work we would do. Its authors recommend pairing AshGC with Sage 2.3.0 rather than swapping charge backends under an older force field |
| `espaloma_charge` | MIT | an alternative AM1-BCC-like GNN; same DGL/PyTorch export problem, less adoption |
| **ONNX Runtime Web** | 1.27.0, MIT | runs an exported GNN on WebGPU or WASM in a tab. Already vendored and working in Molarium for ANI-2x |
| **Dimorphite-DL** | 2.0.2, Apache-2.0 | ordered empirical ionisable-site SMARTS; a possible source for ligand protonation states if we ever want more than the CCD definition |

¹ Figures from the AshGC paper as reported by a literature search, not
re-derived here. Before NAGL ships as a default, the plan runs it against
AmberTools AM1-BCC on our own test ligands and records the difference.

## 5. MD and FFT on WebGPU

| Project | License | State | Gap |
| --- | --- | --- | --- |
| `andeplane/webgpu-md` | — | real browser-side forces, neighbour lists, integration | Lennard-Jones only; no molecular force field, no PME |
| Lexolith (announced 2026-10-12) | — | WebGPU MD of 4096 water molecules with live temperature and energy-drift plots | no protein, no evidence of PME |
| `sameeerkashyap/molframes` | — | WebGPU trajectory viewer | viewer, not an engine |
| Molarium WebGPU path | Apache-2.0 | see §1 | no PBC, no PME, no barostat |
| `larsjoost/wgsl-fft` | MIT | batched 1D FFT, power-of-two plus Bluestein; Rust/wgpu with WGSL in `src/shaders.rs` | no 3D API; Rust host |
| `eugenehp/gpu-fft` | MIT | FFT/IFFT via CubeCL on wgpu | Rust/CubeCL, not standalone WGSL |
| `PracticalXR/minigpu` `gpu_tensor` | MIT | advertises FFT up to 3D | Dart/Flutter; experimental |

**No TypeScript/WGSL 3D FFT library exists.** For PME we write one: three
passes of batched 1D Stockham FFT over a 2/3/5/7-smooth grid (53³–64³ for these
systems) is a known, bounded piece of work, and the MIT Rust kernels above are
readable precedent for the radix stages.

**No browser MD engine anywhere does explicit solvent with PME.** Searched and
not found, as of today.

## 6. Everything else we would otherwise reinvent

| Need | Option | License |
| --- | --- | --- |
| Structure viewer | **3Dmol.js** — same family as the notebook's py3Dmol | BSD-3-Clause |
| | Mol* | MIT |
| | NGL | MIT |
| Force-field data | OpenMM's own `amber14/protein.ff14SB.xml` (224 KB), `amber99sbildn.xml` (474 KB), `amber99sb.xml` (468 KB), `amber14/tip3p.xml`, `amber14/spce.xml`, `tip3p.xml`, `spce.xml`, `hydrogens.xml` (13 KB), and the pre-equilibrated `tip3p.pdb` / `spce.pdb` water boxes (180 KB each) that `addSolvent` tiles — **≈1.2 MB total, ≈1.6 MB with both water boxes**, before gzip | MIT (OpenMM) |
| SMIRNOFF data | `openff-forcefields` `.offxml` | CC BY 4.0 |
| DCD writing | none needed — a 100-line binary writer; the format is a fixed 84-byte header, a title block, an atom count and per-frame coordinate blocks | — |
| AMBER `prmtop` writing | ParmEd is the reference implementation; the format is documented | LGPL — read the spec, do not vendor the code |
| Interaction fingerprints | **ProLIF has no browser equivalent.** Its 1.1.0 interaction definitions (distance and angle criteria per interaction type) and `LigNetwork`'s vis.js layout must be reimplemented and then gated against ProLIF itself | ProLIF is Apache-2.0; its *definitions* are what we port |
| Long-run plumbing | OPFS for checkpoints, File System Access API for streaming large files to disk, Screen Wake Lock to stop the display sleeping, `navigator.storage.estimate()` for quota | platform APIs |

## 7. What this project therefore has to originate

1. **Explicit-solvent PME on WebGPU**: B-spline spreading, a WGSL 3D FFT,
   reciprocal convolution, force interpolation, with OpenMM's α and grid rules.
2. **A neighbour list with periodic boundaries** and a Monte Carlo barostat that
   rescales molecules, in WGSL.
3. **A force-accumulation and position-precision scheme** without f64 and
   without 64-bit atomics (see [PLAN.md](PLAN.md#precision)).
4. **AMBER protein system building in TypeScript**: ffxml parsing, residue
   template matching, `Modeller.addHydrogens` with OpenMM's hydrogen
   definitions and pH variants, `addSolvent` water tiling and ion placement, and
   the PDBFixer steps.
5. **NAGL in ONNX**, exported and then validated against AM1-BCC.
6. **The analyses**: RMSD/RMSF/Rg/2D-RMSD/PCA/cross-correlation and a ProLIF
   equivalent, each with a parity test against pytraj/MDAnalysis/ProLIF.
