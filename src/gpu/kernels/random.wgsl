// Counter-based random numbers: Threefry-2x32-20 and Box-Muller.
//
// The device twin of src/reference/random.ts, and the reason the thermostat can
// be tested at all. A counter-based generator has no state, so the deviate for
// (particle, axis, step) is a pure function of those indices and the seed --
// which means 26,749 threads in no particular order produce exactly what the
// host's sequential loop produces, and a Langevin step with friction switched
// on becomes a differential test instead of a statistical one.
//
// The integer part is bit-identical to the TypeScript, and so is the uniform
// it produces -- deliberately, see `open_unit_interval`. The deviates then
// differ by the accuracy of `log`, `sqrt` and `cos`, which WGSL specifies only
// to a few ulp, so host and device agree on the normals to f32 precision rather
// than exactly. test/random.test.ts covers the integer side; the browser lane
// covers both.
//
// Every word stays u32 and the only arithmetic is add, xor and rotate, all of
// which wrap identically everywhere; there is nothing here for an optimiser to
// reassociate, which is the same property that made the forces integers
// (docs/PRECISION.md).

const KEY_SCHEDULE_PARITY: u32 = 0x1bd11bdau;
const THERMOSTAT_DOMAIN: u32 = 0x9e3779b9u;
const TWO_PI: f32 = 6.283185307179586;

/// Threefry-2x32-20: 64 bits of counter and 64 bits of key to 64 bits out.
///
/// `rotations` and `schedule` are function-scope `var` arrays because WGSL only
/// allows a dynamic index into those; a `const` array needs a constant index.
fn threefry2x32(counter: vec2<u32>, key: vec2<u32>) -> vec2<u32> {
  var rotations = array<u32, 8>(13u, 15u, 26u, 6u, 17u, 29u, 16u, 24u);
  var schedule = array<u32, 3>(
    key.x,
    key.y,
    KEY_SCHEDULE_PARITY ^ key.x ^ key.y,
  );
  var x = counter + key;
  for (var round = 0u; round < 20u; round++) {
    x.x = x.x + x.y;
    let bits = rotations[round % 8u];
    x.y = ((x.y << bits) | (x.y >> (32u - bits))) ^ x.x;
    if ((round + 1u) % 4u == 0u) {
      let injection = (round + 1u) / 4u;
      x.x = x.x + schedule[injection % 3u];
      x.y = x.y + schedule[(injection + 1u) % 3u] + injection;
    }
  }
  return x;
}

/// Thirty-two random bits to a float strictly inside (0, 1).
///
/// The top 23 bits, placed at the midpoint of their bucket so that neither
/// endpoint occurs -- `log` of a zero below would give an infinite deviate.
///
/// Twenty-three and not twenty-four: a midpoint of 2^24 buckets is an odd
/// multiple of 2^-25 and needs a 25-bit mantissa, so at 24 bits every input
/// above 2^23 makes `+ 0.5` a tie, round-to-even merges the buckets in pairs,
/// and the host -- which computes the same thing in f64 -- disagrees. At 23
/// bits each midpoint is an odd multiple of 2^-24, exact in f32, and both sides
/// get the same uniform bit for bit. See src/reference/random.ts for the cost.
fn open_unit_interval(bits: u32) -> f32 {
  return (f32(bits >> 9u) + 0.5) * 1.1920928955078125e-7;  // 2^-23
}

/// Two independent standard normals from one counter.
fn normal_pair(counter: vec2<u32>, key: vec2<u32>) -> vec2<f32> {
  let bits = threefry2x32(counter, key);
  let radius = sqrt(-2.0 * log(open_unit_interval(bits.x)));
  let angle = TWO_PI * open_unit_interval(bits.y);
  return vec2<f32>(radius * cos(angle), radius * sin(angle));
}

/// The three deviates one particle needs this step.
///
/// Two counters give four, and the fourth is dropped. Carrying it into the next
/// particle would be state, and state is what a parallel generator cannot have.
fn thermostat_normals(seed: u32, step: u32, particle: u32) -> vec3<f32> {
  let key = vec2<u32>(seed, THERMOSTAT_DOMAIN);
  let first = normal_pair(vec2<u32>(particle * 2u, step), key);
  let second = normal_pair(vec2<u32>(particle * 2u + 1u, step), key);
  return vec3<f32>(first.x, first.y, second.x);
}
