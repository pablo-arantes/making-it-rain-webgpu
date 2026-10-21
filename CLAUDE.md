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
missing, which looks exactly like an unsupported browser. And **Playwright's
bundled Chromium does not expose WebGPU at all**; the config uses
`channel: "chrome"`, the installed Google Chrome. `tools/chrome.ts` wraps both
facts up for the tools.

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
about 38 minutes at the measured 38 ns/day. It is skipped otherwise.

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

Measured on this machine, 1L2Y through `web/run.html`: **about 208 steps/s**,
5000-step reports roughly every 24 s, so 1 ns at a 2 fs timestep is **~40
minutes** — against the test's 90-minute budget. Minimisation comes first and
shows `progress=0` for its duration, which is about 40 s at the default 10000
steps.

## Capturing measurements

```sh
npm run report:adapter                                    # to stdout
npm run report:adapter -- docs/measurements/device-<x>.json
npm run bench -- docs/measurements/bench-<x>.json
```

Both drive the real page in Chrome. `?bench=1` on the page is what runs the
benchmark, so the benchmark measures exactly what the checks ran on rather than
a separate harness.

Anything in `docs/` that quotes a number should be traceable to a file in
`docs/measurements/`, and `test/units.test.ts` asserts our PME parameters
against the committed OpenMM output, so a drift between the two fails a test
rather than going unnoticed.

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
