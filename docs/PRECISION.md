# Precision

How the engine represents forces and positions without f64 and without 64-bit
atomics, what was tried and abandoned, and the measurements that decided it.

Everything here was measured on an **Apple M3 Pro, Chrome 155 headless,
WebGPU/Metal**, with `npm run test:browser` and
`npm run report:adapter`. The records are
`docs/measurements/device-M3Pro-chrome.json` and
`docs/measurements/bench-M3Pro-chrome.json`. Numbers from one adapter and one
browser build; the point of committing them is that a different device can be
compared with them, not that they generalise.

## Why this needed deciding before any force kernel was written

WebGPU has no f64, no 64-bit atomics, and atomics only on `i32` and `u32`. The
workload makes that matter more than it might sound. For the 3HTB fixture
(26,078 atoms, `docs/measurements/3HTB-JZ4-facts.json`):

| force group | energy (kJ/mol) |
| --- | ---: |
| `NonbondedForce` direct space | **+1,563,444** |
| `NonbondedForce` reciprocal space | **−1,732,547** |
| everything else | +31,075 |
| total | **−138,029** |

Each PME term is more than ten times the answer. A summation that rounds while
accumulating loses the result in that cancellation, and loses a different amount
depending on the order the GPU happened to schedule the threads in.

## What the engine does

**Forces and positions are both 64-bit fixed point**, two 32-bit words, scale
2³², defined once in `src/gpu/kernels/fixed64.wgsl` and mirrored operation for
operation in `src/reference/fixed-point.ts`.

```
value = (f64(i32 high) · 2³² + f64(u32 low)) / 2³²
```

Range ±2³¹, resolution 2⁻³², about 2.3e-10. For forces that is ±2.1e9 kJ/mol/nm
against a largest measured component of 5.6e5 in an unminimised structure, so
about 3800× headroom. For positions it is ±2.1e9 nm against a 6.52 nm box, and
2.3e-10 nm against the 1e-7 nm that the notebook's 1e-6 constraint tolerance
implies on a 0.1 nm X–H bond.

Integer addition is exact, commutative, and cannot be algebraically simplified
by a compiler. All three properties turned out to matter.

Three details are not obvious and each came from a failing test:

1. **Convert a magnitude and negate, never floor a signed value.** For a small
   negative x, `x - floor(x)` is `1 + x`, and an f32 near 1 cannot keep the low
   bits of x. Measured: 2.2e-8 error against 2.3e-10 for the same magnitude
   positive.
2. **Round to nearest, do not truncate.** Truncation is biased toward zero, and
   a bias does not average out.
3. **Convert back through the magnitude too.** A negative value's
   two's-complement form puts a large number in the low word — −1e-9 is
   high = −1, low = 2³² − 4 — and `f32(low)` cannot hold 32 bits. Converting the
   sign-magnitude form keeps the error symmetric, which matters because this
   feeds the integrator's residual, and an asymmetric conversion makes the
   residual sign-dependent, which is a bias rather than noise.

**Positions additionally carry a residual.** The part of each displacement that
did not fit the format is added to the next step's displacement, which bounds
the outstanding error by one unit in the last place however long the run is.

```wgsl
let wanted = dx + residual;
let taken = to_fixed(wanted);
position = add64(position, taken);
residual = wanted - from_fixed(taken);
```

## The measurement that justifies the residual

200,000 displacements of 1e-7 nm onto a 6.5 nm coordinate — 400 ps of one
coordinate at a 2 fs timestep, and the hardest case for a fixed-point
integrator, because one repeated displacement has the *same* conversion error
every step, so it accumulates instead of cancelling.

| scheme | result (nm) | error (nm) |
| --- | ---: | ---: |
| exact (f64) | 6.520000000233722 | — |
| **f32** | **6.5** | **0.02** |
| fixed point, plain | 6.519976869225502 | 2.3e-5 |
| **fixed point, compensated** | **6.520000000251457** | **1.8e-11** |

The compensated integrator is **1,304,247×** better than plain accumulation
here, and the device matched the host reference bit for bit on both.

Three things to read off it. **f32 does not drift, it stops**: the displacement
is below half an ulp of a 6.5 nm f32 coordinate (4.77e-7 nm), so the coordinate
never changes at all and 0.02 nm of motion disappears. **Plain fixed-point
accumulation drifts linearly in the step count** — 2.3e-5 nm over 400 ps is
0.00023 Å and harmless, but it grows with simulation length, and at a
microsecond it would be 0.06 nm. **Residual feedback removes the growth**, not
just the magnitude.

On a realistic history — 200,000 random displacements of ±5e-4 nm, the scale of a
2 fs step — the conversion errors decorrelate and all of it gets easier: f32 is
off by 8.1e-5 nm, plain fixed point by 1.4e-8, compensated by 3.8e-9. The value
of that check is different: it is 200,000 *dependent* integer operations on
which the device still has to agree with the reference exactly, including the
negative branch. It is the check that found the sign asymmetry in item 3 above.

## What was tried and abandoned: double-single positions

The first plan was two f32 words holding a value as their exact unevaluated
sum — Dekker's error-free arithmetic, about 48 bits of mantissa. It is a
standard technique, `src/reference/double-single.ts` implements it correctly, and
the host reference reaches 6.5200000000047 on the test above, 2.3e-10 from the
exact answer.

**On Chrome/Metal the device returns 6.5 and a low word of exactly zero.**

With runtime inputs — nothing const-foldable — both `two_sum` and
`quick_two_sum` return an error term of 0 where it should be 1e-7:

```
 two_sum(runtime).sum   = 6.5
 two_sum(runtime).ERR   = 0          <- should be 1.0000000116860974e-7
 quick_two_sum.sum      = 6.5
 quick_two_sum.ERR      = 0
 200k iters hi          = 6.5
 200k iters lo          = 0
```

The backend simplifies `b - (s - a)` into `b - b`. That identity is true in
exact arithmetic and false in floating point, and it destroys the only thing the
expression exists to compute. Nothing warns; the result is a silently
f32-precision accumulator wearing a double-single type.

So the invariant in `AGENTS.md`: **no scheme may depend on an f32 rounding
surviving the optimiser.** A formulation that happens to survive today would be
broken by a driver update, with no test failing anywhere except the physics.

The Dekker code stays in the tree as a **diagnostic**. `checkDekkerDiagnostic`
runs it on every device, passes whatever it finds, and reports which behaviour
it saw. If some backend computes it properly we will know, and the reason
positions are integers stays measured rather than remembered.

## Primitive throughput

`npm run bench`, Apple M3 Pro, Chrome 155, 1,048,576 contributions:

| | million atomic adds/s |
| --- | ---: |
| fixed-point accumulate, 1 slot | 374.5 |
| fixed-point accumulate, 4096 slots | 551.9 |
| fixed-point accumulate, 65536 slots | 291.3 |

**One slot is faster than 65,536.** Atomic *contention* is not the cost on this
GPU — scattered memory access is. That matters for Phase 1, because the plan
assumed contention would force a gather-based force kernel with a full
neighbour list at 2× the pair work. On this adapter the atomic path looks
affordable. It has to be measured on NVIDIA and on an integrated GPU before the
plan changes, and the determinism invariant is a separate argument for gather
that this measurement does not touch.

The serial position integrator runs at 6.8 million dependent steps per second,
which is a latency baseline rather than anything the engine will do — a real
step integrates every atom in parallel.

## Still open for Phase 1

- Whether PME charge spreading gathers per grid point or uses the atomic path.
  The benchmark above makes the atomic path look better than assumed; the
  determinism invariant still argues for gather. Decide by measuring both.
- Whether the same conclusions hold on NVIDIA, AMD and an integrated GPU, in
  particular whether any backend honours error-free arithmetic.
- A finer position scale, 2⁻⁴⁰ rather than 2⁻³², is available within the same
  two words if the residual ever proves insufficient. It would need the
  decomposition reworked, since the fraction would no longer fit one word.
