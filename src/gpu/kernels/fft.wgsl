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
//   twiddle   exp(sign * 2 pi i * (j % ns) * r / (ns * radix)), from a table
//   scatter   out[base + ((j / ns) * ns * radix + (j % ns) + r * ns) * stride]
//
// and that one `base` formula serves all three axes: stride 1 walks z,
// stride nz walks y, stride ny*nz walks x.

@group(0) @binding(0) var<storage, read>       fft_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> fft_out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform>             stage: Stage;
/// `exp(sign * 2 pi i * j / length)` for j in [0, length), computed on the host
/// in f64 and rounded once to f32. See the note on `twiddle_stride`.
@group(0) @binding(3) var<storage, read>       twiddles: array<vec2<f32>>;

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
  /// 🔴 Appended, not inserted. Adding a field in the middle of this struct
  /// shifts every offset after it and the host packer silently writes the wrong
  /// words -- which has cost this repository time three times (CLAUDE.md).
  ///
  /// `length / (ns * radix)`, the step through the twiddle table for the gather.
  /// `ns * radix` divides `length` at every stage, because it is a prefix
  /// product of the same radices, so this is exact.
  twiddle_stride: u32,
  /// `length / radix`, the step for the butterfly's own twiddle.
  radix_stride: u32,
}

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
      // 🔴 A table lookup rather than `cos`/`sin`, because **WGSL bounds the
      // accuracy of neither**. SwiftShader computes them to 1,587 ulps, which
      // took this transform from 1.4e-6 to 3.3e-4 relative RMS -- measured, in
      // `docs/measurements/fft-twiddle-precision.json`. The angle was never the
      // problem; the transcendental was. `twiddle_index * r < ns * radix`, so
      // the index stays inside the table without a reduction.
      values[r] = cmul(value, twiddles[twiddle_index * r * stage.twiddle_stride]);
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
      // `k * m` can exceed the radix -- 36 for radix 7 -- so it is reduced
      // before scaling, which is the modulo the angle form did implicitly.
      sum += cmul(values[m], twiddles[((k * m) % radix) * stage.radix_stride]);
    }
    fft_out[base + (destination + k * stage.ns) * stage.stride] = sum;
  }
}
