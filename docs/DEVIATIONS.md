# Deviations

Every place the web app does something the notebook does not. The rule from
`AGENTS.md` is that physics never changes silently: each entry says what the
notebook does, what the app does, why, and whether the user is told.

"Shown" means the UI states it at the point where it matters — not buried in a
tooltip — and that `run.json` records it so a run can be reproduced in either
place.

| # | Notebook | App | Why | Shown |
| --- | --- | --- | --- | --- |
| 1 | Equilibration restraints use `!(:H*) & …`, a residue-name mask: 2619 atoms including 1318 hydrogens, histidine unrestrained | **Default** is the heavy-atom mask `!(@H*) & …`: 1311 atoms, no hydrogens, histidine restrained. The notebook's mask stays selectable | The comment, the protocol and the papers all say heavy atoms; the notebook's mask is a cpptraj `:` / `@` slip. Keeping both means published runs stay reproducible | yes — a named choice on the equilibration panel |
| 2 | Preparation is not reproducible: undetermined hydrogens come from Python's unseeded global RNG and are relaxed for 50 iterations; ion placement uses `random.choice`. Three identical runs moved atoms 2.3 Å and the box 15 pm | Every RNG is seeded and the seed is recorded; the same seed gives a byte-identical `system.xml` | A user who cannot rebuild their own system cannot check anything about it | yes — the seed is on the download panel |
| 3 | `LangevinIntegrator`, which has been LFMiddle since OpenMM 8.2 | LFMiddle | Matching what the notebook *runs today*. A notebook run from before OpenMM 8.2 used the old leapfrog Langevin and is **not** reproducible here | yes — the integrator is named in `run.json` and the citation box |
| 4 | PME grid is whatever the platform chose: the Reference platform uses the exact minimum (53³ for 3HTB, 25³ for 1L2Y), CUDA and OpenCL round up to their own FFT-friendly sizes | α from `sqrt(-log(2·tol))/cutoff`, grid rounded up to the nearest 2/3/5/7-smooth size; a run can pin an explicit grid | A WGSL FFT needs a smooth size. The notebook's grid is already platform-dependent, so there is no single value to reproduce | yes — both the minimum and the chosen grid in `run.json` |
| 5 | Ligand default `gaff-2.11`, typed by `antechamber` | Ligand default `openff-2.3.0`; GAFF by upload only | `antechamber` is needed for GAFF **atom typing**, not just charges, and has never been ported to WebAssembly. Sage 2.3.0 is also the release NAGL's charges are fitted alongside | yes — the GAFF options say what they need, and the run records the force field |
| 6 | Ligand charges from AM1-BCC via `sqm` | NAGL (`openff-gnn-am1bcc-1.0.0`) through ONNX | `sqm` cannot run in a tab. Expect ~0.03 e RMSE against the notebook, dominated by the AmberTools-vs-OpenEye difference that already exists in Python | yes — charge method, model version, and the per-ligand comparison when a reference is available |
| 7 | `positions + [30,30,30] Å` before writing `system.pdb` | no shift; the viewer centres instead | A rigid translation under periodic boundaries changes nothing but the coordinates on disk | no — nothing observable changes |
| 8 | pH appears twice: hardcoded `7.4` in the PDBFixer call, and as a field in the topology cell that cannot take effect | One pH field, applied where protonation is actually decided | A control that does nothing is worse than no control | yes — the field says which step it governs |
| 9 | Warns about nothing when the box is too small for the cutoff | Computes the real minimum solute-to-own-image distance after solvation and warns when it is below the cutoff. (1L2Y at the defaults: 0.958 nm against a 1.0 nm cutoff) | `padding < cutoff` is a weak proxy; the geometry is cheap to measure and is what matters | yes — the number and the warning on the topology panel |
| 10 | GROMACS `top`/`gro` from ParmEd, built from a second constraint-free `System` | Best effort, Phase 5, labelled non-authoritative unless it round-trips | ParmEd's GROMACS writer has its own conventions and no step of this protocol reads the format | yes — the download is labelled |
| 11 | Interaction network over at most 10 frames whatever the trajectory length | Every frame, with the frame count stated | A 0.3 frequency threshold over 10 frames is a coarser statement than it looks | yes — the frame count is on the figure |
| 12 | Analysis parameters are retyped into each analysis cell | Everything read from `run.json` | A plot's time axis cannot then disagree with its trajectory | no — it can only remove a class of error |
| 13 | OpenMM's mixed precision keeps positions in f64 and accumulates forces in 64-bit fixed point | Positions in 64-bit fixed point with residual feedback; forces in the same format. There is no f64 and no 64-bit atomic in WebGPU | Measured: 2.3e-10 nm resolution against the 1e-7 nm the constraint tolerance needs, and 1.8e-11 nm accumulated error over 400 ps. The force format is OpenMM's own | no — it is an implementation of the same precision class; `docs/PRECISION.md` has the numbers |

## Not deviations, but worth stating

* **Bitwise agreement across devices is not promised.** Within one device, one
  seed and one workgroup configuration, a run is bit-identical; across GPUs it
  is not, because the kernels are allowed to pick different workgroup shapes.
* **Single precision is the forces' precision.** The engine targets OpenMM's
  mixed-precision behaviour, not its double-precision Reference platform. The
  measured floor for a legitimate single-precision implementation is ~1e-5
  relative force error — OpenMM's own OpenCL platform sits at 1.05e-05 on an
  equilibrated 1L2Y.
* **The notebook remains the oracle.** Anything in this file that turns out to
  change a scientific result, rather than a label or a reproducibility
  property, is a bug in this file's reasoning and should be raised.
