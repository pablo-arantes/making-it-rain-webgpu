/**
 * Counter-based random numbers, identical on the host and on the device.
 *
 * A thermostat needs three normal deviates per moving particle per step, and
 * where they come from decides two things that matter more than speed.
 *
 * **Reproducibility.** A run has to be repeatable from its seed, so nothing
 * here touches `Math.random`. `langevinMiddleStep` refuses to run with friction
 * and no explicit source for exactly that reason.
 *
 * **Testability.** A sequential generator cannot be checked against a parallel
 * one: 26,749 GPU threads have no order to share. A *counter-based* generator
 * has no state at all -- the deviate for (particle, axis, step) is a pure
 * function of those indices and the seed -- so the host and the device compute
 * the same number independently, and the Langevin step becomes differentially
 * testable with friction switched on. Without this, only the friction-zero case
 * could ever be compared, which is the one case the thermostat does nothing in.
 *
 * Threefry-2x32-20, from Random123 (Salmon, Moraes, Dror & Shaw, SC'11): a
 * 20-round Threefish-derived bijection on a 64-bit counter. It is a
 * well-studied generator that passes BigCrush, and it is about twenty lines.
 * `src/gpu/kernels/random.wgsl` is the same algorithm in WGSL, and
 * test/random.test.ts holds them to bit equality.
 *
 * This is bit-compatible with Random123 itself, which is checked rather than
 * asserted: test/random.test.ts runs the nine `threefry2x32` known-answer
 * vectors published in that project's own `tests/kat_vectors`, at 13, 20 and 32
 * rounds. Writing a generator from a remembered specification is exactly the
 * kind of thing that produces a plausible-looking stream with a subtly wrong
 * key schedule, and only a known answer catches it.
 *
 * Note this is *not* OpenMM's stream, and matching it is not possible --
 * OpenMM's GPU platforms generate their own deviates in their own order. Two
 * Langevin trajectories from different noise diverge no matter how exact each
 * step is, so parity with OpenMM is checked at friction zero (exact, step for
 * step) and statistically above it. See docs/DYNAMICS.md.
 */

/** Threefish's key-schedule parity constant, 2^32/phi rounded. */
const KEY_SCHEDULE_PARITY = 0x1bd11bda;

/** Threefry-2x32's rotation constants, cycling every eight rounds. */
const ROTATIONS = [13, 15, 26, 6, 17, 29, 16, 24] as const;

/** The standard round count, and the only one the device compiles. */
export const ROUNDS = 20;

const add32 = (a: number, b: number): number => (a + b) >>> 0;

const rotateLeft = (x: number, bits: number): number =>
  ((x << bits) | (x >>> (32 - bits))) >>> 0;

/**
 * The bijection itself: a 64-bit counter and a 64-bit key to 64 bits of output.
 *
 * Every word is kept in an unsigned 32-bit range throughout. A sum of two
 * such words is below 2^33 and so exact in a double before `>>> 0` reduces it,
 * which is why this needs no `Math.imul` -- there is no multiplication.
 *
 * `rounds` exists for the known-answer vectors, which cover 13 and 32 as well.
 * Nothing but the test passes it: the device compiles 20 and only 20.
 */
export function threefry2x32(
  counter: readonly [number, number],
  key: readonly [number, number],
  rounds: number = ROUNDS,
): [number, number] {
  const schedule = [
    key[0] >>> 0,
    key[1] >>> 0,
    (KEY_SCHEDULE_PARITY ^ key[0] ^ key[1]) >>> 0,
  ];
  let x0 = add32(counter[0], schedule[0] as number);
  let x1 = add32(counter[1], schedule[1] as number);
  for (let round = 0; round < rounds; round++) {
    x0 = add32(x0, x1);
    x1 = rotateLeft(x1, ROTATIONS[round % 8] as number) ^ x0;
    x1 = x1 >>> 0;
    // Every fourth round the key is folded back in, with the injection's own
    // index added to the second word so that no two injections are alike.
    if ((round + 1) % 4 === 0) {
      const injection = (round + 1) / 4;
      x0 = add32(x0, schedule[injection % 3] as number);
      x1 = add32(add32(x1, schedule[(injection + 1) % 3] as number), injection);
    }
  }
  return [x0, x1];
}

/**
 * Thirty-two random bits to a float strictly inside (0, 1).
 *
 * The result is the *midpoint* of its bucket rather than its lower edge, so
 * both endpoints are unreachable. That is not fastidiousness: Box-Muller takes
 * the logarithm of this value, and a zero would produce an infinite deviate.
 *
 * Twenty-three bits, not twenty-four, and the missing one is the point. A
 * midpoint of 2^24 buckets is an odd multiple of 2^-25, which needs 25 bits of
 * mantissa; f32 has 24. Measured: at 24 bits `f32(bits >> 8) + 0.5` is a tie
 * for every input above 2^23, round-to-even collapses the buckets in pairs, and
 * the top half of the interval quietly loses a bit anyway -- while the host,
 * computing the same expression in f64, keeps it and disagrees with the device.
 * At 23 bits every midpoint is an odd multiple of 2^-24, exactly representable,
 * and the two sides produce the same uniform bit for bit.
 *
 * The cost is the tail: the smallest value is 2^-24, so a Box-Muller radius
 * cannot exceed sqrt(48 ln 2) = 5.77 sigma. The probability mass beyond that is
 * 6e-8 per draw and it is not lost, only collapsed onto the boundary radius --
 * the same quantisation every single-precision Box-Muller has, and far below
 * anything a thermostat with friction can register.
 */
export const openUnitInterval = (bits: number): number =>
  ((bits >>> 9) + 0.5) * 2 ** -23;

/** Two independent standard normals from one counter, by Box-Muller. */
export function normalPair(
  counter: readonly [number, number],
  key: readonly [number, number],
): [number, number] {
  const bits = threefry2x32(counter, key);
  const radius = Math.sqrt(-2 * Math.log(openUnitInterval(bits[0])));
  const angle = 2 * Math.PI * openUnitInterval(bits[1]);
  return [radius * Math.cos(angle), radius * Math.sin(angle)];
}

/**
 * A domain constant in the key's second word.
 *
 * It keeps the thermostat's stream distinct from any other use of the same
 * seed -- a barostat's accept/reject draw, or initial velocities -- so that two
 * of them seeded alike do not move in step. 2^32/phi again, as a nothing-up-my-
 * sleeve number.
 */
export const THERMOSTAT_DOMAIN = 0x9e3779b9;

/**
 * The barostat's domain, so its uniforms cannot collide with the thermostat's.
 *
 * A different constant in the same place. Two generators sharing a seed and a
 * counter would produce the same stream, and a barostat whose volume proposals
 * were correlated with the thermostat's noise would sample the wrong ensemble
 * in a way no single-step test could see. 2^32 times the imaginary part of the
 * silver ratio, as another nothing-up-my-sleeve number.
 */
export const BAROSTAT_DOMAIN = 0x6a09e667;

/**
 * The domain for initial velocities.
 *
 * A third constant, for the same reason as the second: the Maxwell-Boltzmann
 * draw that starts a run must not share a stream with the thermostat that
 * continues it. 2^32 times the fractional part of sqrt(3).
 */
export const VELOCITY_DOMAIN = 0xbb67ae85;

/** A standard normal for one particle and axis, on a given domain. */
export function domainNormal(
  domain: number,
  seed: number,
  particle: number,
  axis: number,
): number {
  const half = axis >> 1;
  const pair = normalPair(
    [(particle * 2 + half) >>> 0, seed >>> 0],
    [seed >>> 0, domain],
  );
  return pair[axis & 1] as number;
}

/**
 * The deviate the thermostat uses for one particle, axis and step.
 *
 * Two counters per particle give four deviates, of which three are used. The
 * fourth is discarded rather than carried into the next axis, because a carry
 * is state, and state is what makes a generator unverifiable against a GPU.
 */
export function thermostatNormal(
  seed: number,
  step: number,
  particle: number,
  axis: number,
): number {
  const half = axis >> 1; // axes 0,1 from the first pair; axis 2 from the second
  const pair = normalPair(
    [(particle * 2 + half) >>> 0, step >>> 0],
    [seed >>> 0, THERMOSTAT_DOMAIN],
  );
  return pair[axis & 1] as number;
}

/** A thermostat source bound to a seed and a step, for `langevinMiddleStep`. */
export const thermostatNormals =
  (seed: number, step: number) =>
  (particle: number, axis: number): number =>
    thermostatNormal(seed, step, particle, axis);
