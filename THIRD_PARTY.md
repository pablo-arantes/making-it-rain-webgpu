# Third-party software, data and models

Application code in this repository is MIT, as Making it Rain is. Everything
below is someone else's work, listed with what it is used for and the license it
carries. A bundled file is one the browser downloads; a development dependency
is not shipped.

## Bundled at runtime

Nothing yet beyond the application's own code — Phase 0 ships one page and its
kernels. The entries below are what Phases 1–4 will add, recorded now so the
list is maintained rather than reconstructed.

| Component | Version | Use | License |
| --- | --- | --- | --- |
| OpenMM force-field data — `amber14/protein.ff14SB.xml`, `amber99sbildn.xml`, `amber99sb.xml`, `amber14/tip3p.xml`, `amber14/spce.xml`, `tip3p.xml`, `spce.xml`, `hydrogens.xml`, and the pre-equilibrated `tip3p.pdb` / `spce.pdb` water boxes | OpenMM 8.x | the protein, water and ion parameters, and the water box `addSolvent` tiles — about 1.6 MB | MIT |
| OpenFF force fields — `openff-2.3.0.offxml` and the notebook's `openff-*` / `smirnoff99Frosst-*` list | openff-forcefields | ligand parameters | CC BY 4.0 |
| OpenFF NAGL model — `openff-gnn-am1bcc-1.0.0` (AshGC 1.0), exported to ONNX | 1.0.0 | ligand partial charges | model **CC BY 4.0**; NAGL code MIT |
| RDKit MinimalLib — `@rdkit/rdkit` | 2026.9.1 | SMIRKS matching for SMIRNOFF typing, CCD template matching, ligand perception | BSD-3-Clause |
| ONNX Runtime Web | 1.x | running the NAGL model on WebGPU or WASM | MIT |
| OpenMM compiled to WebAssembly, Reference platform | from OpenMM 8.2 | **tests and the "verify this system" button only** — never dynamics for a user | OpenMM MIT; the Emscripten build scripts and patches are Apache-2.0, © 2026 Rafal P. Wiewiora ([molarium](https://github.com/rafwiewiora/molarium)) |
| 3Dmol.js | 2.x | structure and trajectory viewer | BSD-3-Clause |

## Development only

| Component | Version | License |
| --- | --- | --- |
| TypeScript | 5.9.3 | Apache-2.0 |
| Vite | 8.2.2 | MIT |
| Vitest | 4.1.11 | MIT |
| `@webgpu/types` | 0.1.72 | BSD-3-Clause |
| `webgpu` (Dawn Node bindings) | 0.6.0 | Apache-2.0 / BSD-3-Clause |
| `@playwright/test` | 1.62.1 | Apache-2.0 |
| tsx | 4.20.6 | MIT |

The Python oracle environment (`tools/environment.yml`) is development-only and
is not redistributed: OpenMM (MIT), openmmforcefields (MIT), OpenFF Toolkit
(MIT), PDBFixer (MIT), ParmEd (LGPL-2.1 — used as a tool, its code is not
vendored), RDKit (BSD-3-Clause), AmberTools (GPL-3.0 and the Amber license for
some components — invoked as external programs, never linked or vendored),
pytraj/cpptraj (GPL-3.0), MDAnalysis (GPL-2.0+), ProLIF (Apache-2.0).

🔴 **ParmEd, pytraj and MDAnalysis are copyleft and are used only on the Python
side, as tools.** Their formats are implemented from specifications; their code
is not ported into `src/`.

## Data fetched at the user's request

Fetched from the network only when a user asks for a structure, and never
bundled:

- **RCSB PDB** — `https://files.rcsb.org/download/{id}.pdb`.
- **RCSB Chemical Component Dictionary** —
  `https://files.rcsb.org/ligands/download/{CODE}_ideal.sdf` and `{CODE}.cif`,
  for ligand bond orders and formal charges.

## Studied, not used

Patterns from these informed the engineering approach; no code is taken from
either.

- [AlphaFold2 WebGPU](https://github.com/martin-steinegger/alphafold2-webgpu) —
  **no license file; all rights reserved.** Studied only.
- [LocalFold](https://github.com/sokrypton/localfold) — Beer-ware license
  (Revision 42); permissive in effect, attributed here regardless.

## The protocol

The scientific protocol this implements is
[Making it Rain](https://github.com/pablo-arantes/making-it-rain) (Arantes,
Polêto, Pedebos and Ligabue-Braun,
[JCIM 2021](https://doi.org/10.1021/acs.jcim.1c00998)) and
[Cloud-Bind](https://github.com/pablo-arantes/Cloud-Bind). `reference/Uni_Dock+MD.ipynb`
is included as the specification the port is held to.
