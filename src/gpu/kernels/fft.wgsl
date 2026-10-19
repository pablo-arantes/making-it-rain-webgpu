// One Stockham FFT stage, over every line along one axis of a 3D grid.
//
// Stockham rather than in-place Cooley-Tukey, because Stockham auto-sorts:
// each stage reads one buffer and writes the other with a different access
// pattern, and there is no digit-reversal permutation to get wrong. Two
// buffers and a ping-pong is a small price for deleting a whole class of bug.
//
// `src/reference/fft.ts` is the same algorithm in f64 -- the same stage list,
// the same index arithmetic, the same twiddle angles -- so the two can be
// compared stage by stage. That reference is in turn checked against the direct
// O(n^2) transform in src/reference/dft.ts, which is an independent
// implementation one level further down.
//
// One invocation per butterfly per line. The index arithmetic is
//
//   base    = (line / stride) * (stride * length) + (line % stride)
//   gather    in[base + (j + r * length/radix) * stride]
//   twiddle   exp(sign * 2 pi i * (j % ns) * r / (ns * radix))
//   scatter   out[base + ((j / ns) * ns * radix + (j % ns) + r * ns) * stride]
//
// and that one `base` formula serves all three axes: stride 1 walks z,
// stride nz walks y, stride ny*nz walks x.

@group(0) @binding(0) var<storage, read>       fft_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fft_out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             stage: Stage;

struct Stage {
  length: u32,
  stride: u32,
  line_count: u32,
  radix: u32,
  ns: u32,
  /// -1.0 forward, +1.0 inverse. Neither direction is normalised: PME needs no
  /// normalisation, because the forward and inverse transforms each appear once
  /// in an expression that is quadratic in the grid.
  sign: f32,
}

const TWO_PI: f32 = 6.283185307179586;
const MAX_RADIX: u32 = 7u;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(64)
fn fft_stage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let radix = stage.radix;
  let butterflies = stage.length / radix;
  let total = butterflies * stage.line_count;
  let g = gid.x;
  if (g >= total) { return; }

  let line = g / butterflies;
  let j = g % butterflies;
  let base = (line / stage.stride) * (stage.stride * stage.length)
           + (line % stage.stride);
  let twiddle_index = j % stage.ns;

  // Gather, applying the twiddle on the way in.
  var values: array<vec2<f32>, 7>;
  for (var r = 0u; r < MAX_RADIX; r++) {
    if (r >= radix) { break; }
    let at = base + (j + r * butterflies) * stage.stride;
    let value = fft_in[at];
    if (r == 0u || twiddle_index == 0u) {
      values[r] = value;
    } else {
      let angle = stage.sign * TWO_PI * f32(twiddle_index * r) / f32(stage.ns * radix);
      values[r] = cmul(value, vec2<f32>(cos(angle), sin(angle)));
    }
  }

  // The radix-point transform, by definition. The radix is at most 7, so this
  // is at most 49 terms; a specialised butterfly per radix is an available
  // optimisation, and this is what it would be checked against.
  let destination = (j / stage.ns) * stage.ns * radix + twiddle_index;
  for (var k = 0u; k < MAX_RADIX; k++) {
    if (k >= radix) { break; }
    var sum = vec2<f32>(0.0, 0.0);
    for (var m = 0u; m < MAX_RADIX; m++) {
      if (m >= radix) { break; }
      let angle = stage.sign * TWO_PI * f32(k * m) / f32(radix);
      sum += cmul(values[m], vec2<f32>(cos(angle), sin(angle)));
    }
    fft_out[base + (destination + k * stage.ns) * stage.stride] = sum;
  }
}
