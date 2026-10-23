# Working on this checkout

`AGENTS.md` has the invariants. This file is the operational half: how to run
things, and the traps that have already cost time.

## Install

```sh
npm install
```

🔴 **If npm fails with `EACCES` on `~/.npm/_cacache`**, the cache has
root-owned files from an earlier `sudo npm`. Use a different cache rather than
`sudo chown` on a home directory:

```sh
npm install --cache /tmp/mir-npm-cache
```

🔴 **`package.json`'s `allowScripts` block is load-bearing.** npm 11 blocks
install scripts by default, and `esbuild`'s and `webgpu`'s postinstall scripts
are what fetch their platform binaries. Without them Vite and Vitest do not run
at all. The block is the documented way to approve them.

The Python oracle is separate and is described in `tools/README.md`.

## The five lanes

| Lane | Command | Needs |
| --- | --- | --- |
| Typecheck | `npm run build` | nothing |
| Unit + reference | `npm test` | nothing |
| Slow reference | `npm run test:slow` | nothing, and a few minutes |
| GPU, through Dawn | `npm run test:gpu` | a loadable `webgpu` binary — **not this machine**, see below |
| GPU, through Chrome | `npm run test:browser` | Google Chrome installed, and ~9 minutes |
| OpenMM oracle | `npm run test:oracle` | `vendor/openmm-wasm/`, which Phase 1 builds |
| Benchmark | `npm run bench` | Google Chrome installed |
| Everything | `npm run check` | nothing |

`npm run check` is typecheck, unit tests and the web build. It deliberately does
not include a GPU lane, so it stays runnable in CI on a machine with no GPU.

`*.slow.test.ts` is excluded from `npm test` and collected by
`vitest.slow.config.ts` instead. Not lower-value tests — the minimiser's parity
against OpenMM is one — but minutes of double-precision PME, because a minimiser
calls the force field fifty times and the reference evaluates the mesh in f64 on
the CPU. A unit suite that takes minutes stops being run, which is worse than a
lane that is explicitly separate.

## Reaching a GPU

🔴 **No published Dawn binary loads on macOS 14.** Every `webgpu` version from
0.3.10 to 0.6.0 fails with

```
Symbol not found: _OBJC_CLASS_$_MTLLogStateDescriptor
  (built for macOS 26.0 which is newer than running OS)
```

`MTLLogStateDescriptor` arrived in macOS 15. So on this machine
`npm run test:gpu` reports itself unavailable with that reason, and
`npm run test:browser` is the lane that actually runs the kernels. The Dawn lane
is kept because it works in CI on Linux and will work here after a macOS
upgrade. Both lanes call the same functions in `src/gpu/checks.ts`, so neither
can pass something the other would fail.

🔴 **Two things about getting WebGPU in a browser from a script.**
`navigator.gpu` exists only in a **secure context**, so a page must be served
over `http://localhost` — `about:blank` and `file://` both report WebGPU as
missing, which looks exactly like an unsupported browser. And Playwright's
bundled Chromium gives you a **different adapter**: the config uses
`channel: "chrome"`, the installed Google Chrome, because that is where the
Metal path is. `tools/chrome.ts` wraps both facts up for the tools.

🔴 **An earlier version of this said the bundled Chromium "does not expose
WebGPU at all". It does** — as SwiftShader, `vendor: "google"`,
`architecture: "swiftshader"`, `isFallback: true` — and that wrong claim is why
nobody had looked at it for two phases. It is the only second WebGPU
implementation available on a one-GPU machine, so it is worth a great deal:

```sh
MIR_CHROME_CHANNEL=chromium npm run report:adapter
```

Measured there, in `docs/measurements/adapter-swiftshader-m3pro.json`: all four
fixed-point checks **pass**, so the integer-position design is not an Apple
workaround that happens to work. Dekker error-free arithmetic **works** on this
adapter, where Chrome/Metal folds it to zero — the counterexample that makes
that a device property rather than a WebGPU one, and the reason fixed point was
the right call either way. And the mixed-radix FFT **used to fail its 1e-5
threshold there at 3.27e-4**, 232× worse than Metal's 1.41e-6 — which is what
found the twiddle bug below. Both adapters now read about 1.56e-7.

🔴 **That was the backend's `sin` and `cos`, measured rather than suspected,
and the kernel no longer calls them.** A seventh device check probes them at the FFT's own twiddle
arguments and separates the two candidates: SwiftShader's are wrong by 1.89e-4,
**1,587 ulps** of f32, where Metal's sit within **0.77 of one ulp** — and the
f32 rounding of the angle, the other suspect, is the same on both (5.95e-7
against 7.21e-7 radians). WGSL bounds neither function, so that backend is
conforming and the kernel is the thing assuming too much.
`docs/measurements/fft-twiddle-precision.json` has the worst case in full.

The fix was a host-computed twiddle table, f64 rounded once to f32, indexed by
integer arithmetic — and it took SwiftShader from 3.27e-4 to **1.567e-7** and
Metal from 1.41e-6 to **1.561e-7**, so the two adapters now agree to 0.4%. 🔴
Metal improving by 9× was not predicted: the angle's own f32 rounding, which the
diagnosis had exonerated for the *discrepancy*, was the dominant term on the
adapter whose `cos` was already good. 🔴 And it is **not** an engine accuracy
win — the force error moved only 0.32%, because the f32 spread and interpolation
either side of the transform dominate it. The lesson to keep: an intrinsic WGSL
does not bound has no place in a numerical kernel, whatever one device does
with it.

## The pages

| Page | What it is |
| --- | --- |
| `web/index.html` | the GPU panel and the precision checks |
| `web/prepare.html` | a PDB ID or an upload in, an OpenMM `System` out; a ligand needs its parameters uploaded as an ffxml |
| `web/run.html` | upload a system, verify it, run it, download the artefacts |
| `web/analyse.html` | a DCD and its topology in, the notebook's figures out, protein and ligand |
| `web/validate.html` | every differential check against OpenMM, development only |

All five are named inputs in `vite.config.ts`. 🔴 Vite discovers only
`index.html` by itself, so a new page that is not listed there builds in
development and is **silently missing from a production build**.

`MIR_LONG_RUN=1 npx playwright test -g nanosecond` runs the plan's 1 ns gate,
**about 8 minutes** at the measured 210 ns/day. It is skipped otherwise. It was
43 minutes before the kernel work in `docs/measurements/profile-step-m3pro.json`,
so treat any older figure for it with suspicion.

🔴 **Do not edit `src/` while a browser run is in flight.** The dev server
watches it, and `web/run.ts` imports the engine straight out of it — `run.ts` →
`engine/protocol.ts` → `analysis/pdb.ts`, for one real example. Touching
anything in that graph invalidates the module, and with no HMR handlers Vite
falls back to a **full page reload**, which drops the uploaded files and the
form. `playwright.config.ts` serves the long gate from `vite preview` instead,
which has no watcher; the ordinary `npm run test:browser` lane still uses the
dev server.

🔴 **Kill stray browsers before running `npm test`.** Playwright's
`reuseExistingServer` leaves `npm run dev` behind, and an interrupted browser
lane leaves Chrome behind with it. With 63 of those processes alive, the three
3HTB cases in `test/nonbonded-reference.test.ts` — 4.4 s each on an idle
machine, against a 60 s budget — took the file past 112 s and one of them
timed out. The report names the test, which is a numerical parity check against
OpenMM, so it reads as a physics regression rather than as contention: nothing
in the output says "timeout" unless you look for the absence of an
`AssertionError`. `pkill -f "Google Chrome"; pkill -f vite` first, and suspect
this whenever a test that is deterministic by construction fails once.

🔴 **`hydrogens.xml` has a `UNK` entry, and the notebook names every ligand
`UNK`.** That entry is an unknown *amino acid* — hydrogens on N, CA, CB, CG. So
a ligand called `UNK` looks like a residue whose hydrogens can be rebuilt: the
preparation path strips all twelve of JZ4's, then adds none back because the
ligand has no backbone for those definitions to attach to, and template
matching fails on an element signature that is short by twelve hydrogens.
`web/prepare.html` asks which residue is the ligand rather than inferring it,
and that residue is excluded from both hydrogen stripping and heterogen
dropping. `rebuildableHydrogen` in `src/prep/hydrogens.ts` is the general rule:
never remove what you cannot put back.

🔴 **A ligand's bonds live in the PDB's `CONECT` records and nowhere else.**
`residues.xml` has no entry for a ligand, so `inferBonds` finds nothing for it.
`addHydrogens` used to return `conect: []`, which lost them — and the symptom is
remote from the cause: the ligand keeps every atom, its element signature
matches its template exactly, and matching fails on the bond graph because the
residue has no bonds at all.

🔴 **`adapter.limits`, `device.limits` and *requested* limits are three
different things, and the gap is large.** On this M3 Pro the adapter reports
4.29 GB of `maxStorageBufferBindingSize`; a device created without asking for
anything gets WebGPU's default of **128 MB**; the engine asks for more and gets
it. A preflight reading the adapter's maxima claimed 4,473,924 atoms and one
reading an unrequested device's claimed 139,810 — same GPU, a 32× spread.
Anything deciding what a run can do must read the limits of the device the
engine will actually use. `deviceFitness` in `src/gpu/device.ts` takes a plain
limits record for exactly this reason, so the caller has to be explicit about
which it passes, and `DEVICE_PROFILES` lets the arithmetic be tested against
other classes of hardware from one machine.

🔴 **Do not read renderer CPU as a progress signal.** A healthy 1 ns run sits at
about 1% renderer CPU and accumulates a few seconds over many minutes — the work
is on the GPU and the loop batches steps per submission. A stalled page looks
identical. One gate was killed early on exactly this misreading. The real signal
is `window.__mirRun` and the progress element:

```js
await page.evaluate(() => window.__mirRun)   // state, stages, reports, artefacts
await page.locator("progress").getAttribute("value")   // steps completed
```

Measured on this machine, 1L2Y through `web/run.html`: **about 1,200 steps/s**,
so 1 ns at a 2 fs timestep is **under 8 minutes** — against the test's
90-minute budget. Minimisation comes first and shows `progress=0` for its
duration. 3HTB, at 26,749 atoms, runs at about **150 steps/s** (26 ns/day).

🔴 **Read `test/browser/watch.ts`'s ticker rather than theorising from Chrome's
stderr.** Both long specs install it — the 1 ns gate and `bonded.spec.ts` — and
it samples the page once a minute, printing from a `finally` so the output
survives the failure. One probe serves both pages: `validate.html` sets a
`__mir*` global per check as it finishes, so the count and names *are* its
progress, while `run.html` has a step count and `__mirRun.state`.

```
15:34:31 3HTB heap=91MB done=12[__mirCellList __mirConstraints __mirEngineError …]
20:15:53 gate heap=12MB state=running steps=325000 done=1[__mirRun]
```

It also catches `crash` and `close`, the two events that say the page went away
rather than merely erred and the two a spec's own console handlers never see.
That is all there because four runs have failed with

```
Error: page.waitForFunction: Target page, context or browser has been closed
```

which names the wait, so it reads as a condition that never came true — a
stalled kernel — when the page had simply gone away. Twice on the gate and twice
on `bonded.spec.ts`'s 3HTB case, which now stands at three passes to two
failures. **Three explanations were offered and all three were wrong**, each
eliminated only once there was a measurement:

| theory | what killed it |
| --- | --- |
| a rebuild of `dist/` under the live `vite preview` | the browser lane runs off the **dev server**, never touches `dist/`, and failed identically |
| display sleep | `caffeinate -dimsu` made it fail *twelve times sooner* (5.4 min against 66), and `CVDisplayLink ... CVReturn: -6670` appears in runs that **pass** |
| renderer OOM | the heap is flat at 11–18 MB across 15 minutes with no trend |

The instrumented gate passed in 15.3 minutes, stepping monotonically to 490,000
at a steady 554 steps/s, and 3HTB has passed instrumented too, its check list
growing 6, 8, 12, 12, 14 with the heap between 52 and 91 MB. The one thing that
changed is the 60-second `page.evaluate`, which also touches an otherwise
un-interacted tab — a **hypothesis**, not a finding, and exactly the kind of
coincidence that produced the three rows above. The failures stay unexplained.

🔴 The practical rule is the instrumentation rather than any theory: a page that
dies with a rising step count or a growing check list is an environment problem,
and one that dies with either flat is yours. And verify a watcher fires before
trusting it — 1L2Y finishes both its waits inside the interval and prints
nothing, which is indistinguishable from a watcher that never works.

🔴 **Those numbers moved by 6x in one day, so do not trust a remembered one.**
`docs/measurements/profile-step-m3pro.json` and `gate-1ns-m3pro.json` carry the
current ones and what changed. The engine was at 208 steps/s on 1L2Y and 82 on
3HTB before the pair kernel's shell split, the loop-free minimum image, the
position fold and the neighbour list.

## Capturing measurements

```sh
npm run report:adapter                                    # to stdout
npm run report:adapter -- docs/measurements/device-<x>.json
npm run bench -- docs/measurements/bench-<x>.json
npm run report:engine-error -- docs/measurements/engine-force-error-equilibrated-m3pro.json
npm run report:verify -- 1L2Y                             # ~2 s, any fixture
npm run report:verify -- 1L2Y-dodecahedron
```

🔴 **Run `report:engine-error` after touching a kernel.** It is the figure
`AGENTS.md` cites for where the engine sits against OpenMM, it is measured at
the one saved frame the thresholds are defined at, and it was produced by hand
once — after which four kernel changes went past it unverified. A passing
differential threshold is not the same evidence: re-measured after that work the
figure is bit-identical to all 17 digits, which says the changes were *exact*,
and a threshold cannot show that.

Both drive the real page in Chrome. `?bench=1` on the page is what runs the
benchmark, so the benchmark measures exactly what the checks ran on rather than
a separate harness.

Anything in `docs/` that quotes a number should be traceable to a file in
`docs/measurements/`, and `test/units.test.ts` asserts our PME parameters
against the committed OpenMM output, so a drift between the two fails a test
rather than going unnoticed.

## A tight GPU loop

🔴 **Do not use `npm run test:browser` while iterating.** It is 6.5 minutes and
5.9 of those are `bonded.spec.ts`, the 3HTB parity, which cannot tell you
anything about the kernel you just changed. Measured alternatives:

| | |
| --- | --- |
| `npm run report:verify -- <fixture>` | **~2 s** — does the GPU agree with the reference |
| `npx playwright test test/browser/checks.spec.ts` | 1.9 s |
| `npx playwright test test/browser/profile.spec.ts` | 23 s, includes the GPU checks |
| `npx playwright test --ui` | watch mode, re-runs on save |
| `npm run test:browser` | 6.5 min — for the end, not the loop |

`--ui` is what keeps a browser and a dev server up across edits. It needs
`channel: "chrome"`, which `playwright.config.ts` already sets: Playwright's
bundled Chromium is SwiftShader rather than Metal (see above), so a check that
passes there is not evidence about this GPU and one that fails there may be
about the adapter.

🔴 **`web/validate.html` runs every check in sequence, and the late ones are the
slow ones** — so waiting on a late global pays for all the earlier ones, which
on 3HTB is minutes an iteration. `?only=` fixes that:

```
/validate.html?fixture=1L2Y-dodecahedron&only=verify
/validate.html?fixture=3HTB&only=neighbours,cells
```

Names: `bonded`, `constraints`, `cells`, `verify`, `engine-error`,
`neighbours`, `profile`, `random`, and `dynamics` for the whole interdependent
tail (integrators, loop, minimiser, restraints, persistence, PBC, protocol,
barostat) — those share fixtures and devices, so they are one unit rather than
eight. A skipped check records `{ skipped: true }` instead of staying
undefined, so a misspelled filter resolves rather than hanging a probe for
fifteen minutes.

**Before any browser run**: `pkill -f "Google Chrome"; pkill -f vite`. And
build the fixtures once up front with the **oracle** environment, so the loop
never waits on OpenMM:

```sh
~/miniforge3/envs/parametrizani_env/bin/python -I tools/build_triclinic_fixture.py \
    --shape dodecahedron --out fixtures/1L2Y-dodecahedron
```

## Findings, in the order they would cost time again

🔴 **Chrome/Metal folds Dekker error-free arithmetic to zero.** `two_sum`'s
error term comes back as exactly 0 for runtime inputs. This silently turns a
double-single accumulator into an f32 one. It is why positions are fixed-point
integers. Full measurement in `docs/PRECISION.md`; a diagnostic check reports
what each device does, and it is expected to "fail" (report folding) on Apple.

🔴 **Truncation in a fixed-point conversion is a physics bug, not a rounding
detail.** It biases every conversion the same way, so a repeated displacement
drifts linearly in the step count: 2.3e-5 nm over 400 ps, growing to 0.06 nm at
a microsecond. Rounding to nearest is necessary and not sufficient — the
integrator also carries the residual. `roundTiesToEven` in the reference exists
because WGSL's `round` resolves ties to even and `Math.round` resolves them
upward, and the two sides have to agree bit for bit.

🔴 **Test both signs.** Two of the three numerical bugs in Phase 0 were sign
asymmetries, and in both cases the positive-only test passed: flooring a signed
value loses the low bits of a small negative, and `f32(low)` cannot hold the
large low word a negative's two's-complement form produces. The random-walk
check found the second one; the stagnation check, with only positive
displacements, did not.

🔴 **Atomic contention is not the cost on Apple.** 1 accumulator slot runs at
374 M adds/s and 65,536 slots at 291 M/s — scattered memory is slower than
contention. The plan assumed the opposite and budgeted a gather-based force
kernel at 2× the pair work. Do not act on this until it is measured on NVIDIA
and on an integrated GPU; the determinism invariant is a separate argument for
gather.

🔴 **A throughput number from an unminimised fixture measures a diverging
run.** 3HTB's fixture state is not relaxed, and 200 steps of it go from -7.6
kJ/mol an atom to -26,893. A diverging run is *slower* — its atoms spread, its
cells stop being even — so it reads as a slow engine rather than a broken run,
and two such numbers happily "compare". 1L2Y's fixture is relaxed, which is why
it never showed this. `validateProfile` minimises first and reports the final
energy per atom of both paths; if those two disagree, the comparison is
meaningless whatever the times say.

🔴 **The 27-cell shell only reaches one cell width.** So a search radius larger
than the cell leaves candidates two cells out that the scan never visits. The
cells are therefore sized by `cutoff + skin`, not by the cutoff — with
cutoff-sized cells, 3HTB's are 1.096 nm and a list built to 1.1 nm silently
lost every pair between the two, which came apart a few hundred steps later.
**1L2Y cannot test this**: three cells an axis means the shell *is* the whole
box, so nothing is ever out of reach and every check passes at any radius.
`validateNeighbours` is the check that can, and it holds the list and the shell
to bit-for-bit agreement — 0 of 80,247 force components differing — because a
lost pair is a discontinuity in the force rather than something a tolerance
would catch.

🔴 **`layout: "auto"` derives a bind group layout per *entry point*, from the
bindings that entry point statically uses.** So a pass must be handed exactly
those and no more: `direct_pairs` reads `neighbour_count` but never
`neighbours`, and passing both is rejected outright. One shared bindings array
for several entry points of the same module does not work, and the error names
the entry point rather than the cause.

🔴 **A `GPUQuerySet` holds 4096 timestamps, and the profiler needs two per pass
per repeat.** Adding eight passes to the step took 3HTB past it at 40 repeats,
and the failure surfaced as `Invalid QuerySet` on an unrelated later pass rather
than where the set was made. `profilePasses` clamps `repeats` to fit and reports
what it actually used, so a caller dividing by the number it asked for does not
quietly report times that are too small.

🔴 **Profile the big fixture, not the small one.** 1L2Y is 2,373 atoms and
3HTB is 26,749, and for a long time only 1L2Y had a measured speed. A step cost
4.8 ms on 1L2Y and **629 ms** on 3HTB — 11× the atoms and 130× the time, with
the same candidates per atom. Two separate defects were hiding in that gap and
neither showed up on 1L2Y at all: one of the fixes moved 1L2Y by 2% and 3HTB by
18×. `npm run bench` and the 1 ns gate both use 1L2Y, so neither would have
caught it.

🔴 **A loop in the pair kernel's innermost block costs 18×, however few times
it runs.** `minimum_image64` subtracted one box at a time under a `for` bounded
at 64. For a pair inside the cutoff it ran *zero* times, so counting iterations
said it was free — and that reasoning is why it survived this long. The cost is
not the iterations, it is that a nested loop stops the backend unrolling and
pipelining the candidate loop around it. Making it loop-free took 3HTB's pair
kernel from 437,782 µs a step to 27,319. Treat any loop, or any nested
conditional block, inside that kernel as expensive until measured otherwise:
putting the general wrap back behind an integer `|d| < box` guard, so the block
almost never ran, bought **nothing** (72,171 → 72,688 µs).

🔴 **One invocation per atom is not enough threads.** The pair kernel looped
over the 27 cells of the neighbour shell, which left `particleCount` threads in
flight — 2,373 on 1L2Y, 38 workgroups of 64. One invocation per (atom, cell)
is the same work with 27× the threads and was worth 2.4× on 1L2Y. It is exact
because the force accumulators are fixed-point atomics, so partial sums
recombine bit for bit; `docs/measurements/profile-step-m3pro.json` has it.

🔴 **A cheap pass's timestamp is contaminated by an expensive one beside it.**
In the same submission as a 629 ms pair kernel, `cell/clear` — which clears a
216-entry array — reported 15,383 µs; next to a 437 ms one it reported 852 µs,
and with the kernel fixed, 6.7 µs. The pair kernel's own number was reproducible
to 0.1% across runs, so read the big passes and ignore the small ones whenever
one pass dominates the submission.

🔴 **The pair kernel reads *wrapped* positions, binding 0 of its bind group.**
`cell/wrap` folds every coordinate into `[0, box)` once per step, which is what
makes `nearest_image_box`'s single conditional shift provably the whole minimum
image. Two consequences. A new path that encodes the pair pass has to encode the
fold first, or the kernel differences a buffer of zeros — a wrong answer, not an
error; `planCellList` does it, and the host-supplied-list branch in
`nonbonded.ts` has to do it by hand. And `axis_cell` shares `wrap_axis` with the
fold on purpose: if the home cell and the pair vector disagreed for an atom near
a boundary, atom *i* would miss *j* while *j* still found *i*, which leaks
momentum.

🔴 **For a triclinic cell the oracle is OpenMM's CPU platform, not Reference.**
The Reference platform's neighbour list reports pairs that are genuinely inside
the cutoff as non-interacting, and the loss scales with how skewed the cell is —
same solute, same padding, same 1.0 nm cutoff:

| shape | missed | narrowest ÷ shortest edge | closest missed |
| --- | --- | --- | --- |
| cube | 0 of 850 | 1.000 | — |
| octahedron | 23 of 657 | 0.816 | 0.7156 nm |
| dodecahedron | 72 of 689 | 0.707 | 0.6394 nm |

Everywhere else in this project Reference *is* the oracle, which is what makes
this a trap: the minimum image, the cell list, alpha, the dispersion correction,
the exceptions and the exclusion correction were each eliminated before the
oracle came under suspicion.

🔴 **The missed pairs are not confined to the cutoff**, and an earlier version
of this note said they were. The closest one is at 0.64 nm against a 1.0 nm
cutoff, where a Coulomb or Lennard-Jones term is not small. What keeps the total
error at a few kJ/mol is cancellation — 7.0 low on Lennard-Jones against 11.0
high on Coulomb — not the individual pairs being negligible. So do not assume a
small total means a small defect. `docs/measurements/
triclinic-reference-parity.json` has the whole chain.

🔴 **Build triclinic fixtures with the oracle environment, OpenMM 8.4+.**
`addSolvent(boxShape="dodecahedron")` on OpenMM 7.7 removes clashes across a
*rectangular* boundary: atoms end up 0.0168 nm apart and the potential energy is
9.6e10 kJ/mol. The cube is unaffected, so nothing warns you — the fixture simply
encodes a structure no force field can describe. `~/miniforge3/envs/openmm` is
7.7 and is **not** the oracle; `docs/measurements/README.md` names the one that
is.

🔴 **A dodecahedron needs far more padding than a cube, and the refusal is
geometric rather than a missing feature.** The 27-cell scan needs three cells an
axis measured across the *perpendicular* widths, and a rhombic dodecahedron's
narrowest width is 0.707 of its edge — so it needs `d >= 3*sqrt(2)*cutoff`,
4.24 nm at a 1 nm cutoff against the cube's 3.0. 1L2Y at the notebook's 0.4 nm
padding gives 2x2x2 cells and is refused; at 1.5 nm padding it is 4.25 nm wide,
gives 3x3x3, and runs. Expect roughly 5,000 atoms where the cube needs 2,400 —
the volume saving per unit padding is real, but the *minimum* padding is larger,
so a small solute may be better off in a cube.

🔴 **A triclinic cell's cutoff is bounded by its narrowest *perpendicular*
width, not by any edge length.** For a 3 nm dodecahedron the edges are 3, 3 and
3.67 nm while the widths are 2.12, 2.12 and 3 — so an edge-based bound passes a
1.5 nm cutoff that silently loses pairs, where the real ceiling is 1.06. The
notebook's default 1.0 nm fits a 3 nm dodecahedron, but only just.

Within that bound the minimum image is three subtractions, the same shape as the
rectangular path. Beyond it the reduction starts returning the wrong image, and
`src/reference/triclinic.ts` has the measurement: 0 wrong of 34,992 in-cutoff
pairs at 1.0 nm, 6,839 of 91,663 at 1.3× the bound. An earlier reading of the
same arithmetic concluded a 27-image search was unavoidable — that was measured
over *all* separations instead of the in-cutoff ones, and was wrong.

🔴 **A neighbour list is correct when it is built, whatever its rebuild
criterion.** So comparing forces at one configuration cannot test the criterion
at all — what a wrong one costs is pairs that go missing *between* rebuilds,
intermittently. `validateNeighbours` therefore runs both paths NVE from the same
minimised state and compares the energy slopes: equal to seven significant
figures over 248 rebuilds on 1L2Y and 63 on 3HTB, so the list reproduces the
cell-shell trajectory bit for bit. For scale, a stale cell list at an interval
of 20 steps reads 13.7 kJ/mol/ns/atom against 0.0023 rebuilt every step.

That check also has to prove it is comparing two things: the list run must
rebuild and the shell run must not, or two equal slopes are one path agreeing
with itself. Those counts are asserted. Treat the slope itself as a comparison
and never as the engine's drift figure — it is NVE from an unequilibrated
structure, and the calibrated gate is `validate-loop.ts`.

🔴 **The first non-Apple results, from CI's first run, and they are better
than expected.** Two implementations beyond Metal have now executed this code,
and neither is Apple hardware:

| where | what happened |
| --- | --- |
| Chrome + Mesa software adapter, `ubuntu-latest` | **all seven precision checks passed**, `checked.ok` true, `maxAtoms` sane. The only failure was `expect(adapter.isFallback).toBe(false)` — an assertion about *hardware*, now local-only |
| Dawn + lavapipe, `ubuntu-latest` | got an adapter (lavapipe's own "maxDynamicUniformBuffersPerPipelineLayout artificially reduced" warnings are in the log), then the **worker crashed**: `[vitest-pool]: Worker forks emitted error / Worker exited unexpectedly`, no assertion reached |

The passing column is the one to notice. It includes the mixed-radix FFT, which
**failed at 3.3e-4 on SwiftShader until the twiddle table landed** — so that fix
is what made a software adapter viable, and the engine's precision no longer
depends on the driver's `sin` and `cos`. The fixed-point design carries across
both, as the SwiftShader probe already suggested.

🔴 The Dawn crash is **unexplained and worth chasing**, and the honest first
question is whether it is lavapipe on a two-core runner or one of our kernels
taking the process down. It needs a stack, not a conclusion. Until then the
`gpu` job is `continue-on-error` again — a native crash is not something
`skipIf` can report, so the lane cannot self-report there, which is the flaw in
the reasoning that made it required. That reasoning was measured on macOS, where
Dawn cannot load at all and the failure is a catchable `dlopen` error; on Linux
it loads and the process dies instead.

Also from that run: `npm ci` works on the runner, `check` passes, and the
`oracle` job cannot build its environment — `mamba` exits 1 resolving
`tools/environment.yml` on linux-64, which is a packaging problem and not a
physics one.

🔴 **This machine cannot catch a binding overrun.**
`maxStorageBuffersPerShaderStage` has a mandated floor of 8 and Apple reports
far more, so a kernel that bound nine would compile here, pass every
differential check here, and fail to create its pipeline on a conforming Intel
or NVIDIA device — first seen as a user's bug report.
`test/kernel-bindings.test.ts` follows each entry point's calls through the
*composed* source and asserts the count in the no-GPU lane. The busiest is
`direct_pairs_list` at 7 of 8, so there is exactly one spare.

🔴 **`GPUBufferUsage` at module scope breaks the no-GPU lane.** It is a global
that exists only where WebGPU does, so `const STORAGE = GPUBufferUsage.STORAGE`
at the top of a file throws when the *module loads* — not when a buffer is made.
One constant added to `src/gpu/forces.ts` took out two unrelated test files,
because `checks.ts` imports `pme.ts` imports `forces.ts`, and the error named the
constant rather than the import that triggered it. Use the functions in
`src/gpu/usage.ts`; `test/module-load.test.ts` imports every module under
`src/gpu` in the no-GPU lane and is what keeps this from coming back.

🔴 **A WGSL validation error is a console warning and a wrong answer**, which a
differential test cannot distinguish from a bad kernel. `dispatch()` in
`src/gpu/compute.ts` pushes an error scope and reads compilation messages for
exactly that reason. Do not add a dispatch path that skips it.

## Layout

```
src/engine/      units, constants, System/State, the run loop and the protocol
src/gpu/         device, compute helper, kernel composition, checks, bench
src/gpu/kernels/ WGSL; fixed64.wgsl is shared and prepended by composeKernel
src/reference/   independent implementations the kernels are tested against;
                 never imports from src/gpu/
src/prep/        PDB -> System: ffxml, templates, hydrogens, solvation, prmtop
src/analysis/    DCD read/write, PDB read/write, NPY, logs, protein analyses
web/             the five pages above
test/            Vitest; *.gpu.test.ts is the Dawn lane, *.oracle.test.ts the
                 OpenMM one, both opt-in and both self-reporting when skipped
test/browser/    Playwright
tools/           Python oracle and probes, plus the Chrome-driven TS tools
docs/            PLAN, NOTEBOOK_PARITY, PRIOR_ART, PRECISION, DEVIATIONS,
                 measurements
```

WGSL has no include directive. `composeKernel` in `src/gpu/kernels.ts` prepends
shared source, so there is exactly one copy of the fixed-point arithmetic rather
than a copy per kernel that can drift from the reference.
