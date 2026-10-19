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
| GPU, through Dawn | `npm run test:gpu` | a loadable `webgpu` binary — **not this machine**, see below |
| GPU, through Chrome | `npm run test:browser` | Google Chrome installed |
| OpenMM oracle | `npm run test:oracle` | `vendor/openmm-wasm/`, which Phase 1 builds |
| Benchmark | `npm run bench` | Google Chrome installed |
| Everything | `npm run check` | nothing |

`npm run check` is typecheck, unit tests and the web build. It deliberately does
not include a GPU lane, so it stays runnable in CI on a machine with no GPU.

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

🔴 **A WGSL validation error is a console warning and a wrong answer**, which a
differential test cannot distinguish from a bad kernel. `dispatch()` in
`src/gpu/compute.ts` pushes an error scope and reads compilation messages for
exactly that reason. Do not add a dispatch path that skips it.

## Layout

```
src/engine/      units and constants; Phase 1 adds System/State and the run loop
src/gpu/         device, compute helper, kernel composition, checks, bench
src/gpu/kernels/ WGSL; fixed64.wgsl is shared and prepended by composeKernel
src/reference/   independent implementations the kernels are tested against;
                 never imports from src/gpu/
src/prep/        empty until Phase 3
src/analysis/    empty until Phase 3
web/             the page
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
