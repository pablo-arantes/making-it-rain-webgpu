/**
 * What this device's `sin` and `cos` actually do, at the FFT's own arguments.
 *
 * 🔴 WGSL leaves the accuracy of `sin` and `cos` **implementation-defined**.
 * There is no ULP bound in the specification, so a backend may reduce the
 * argument badly, or evaluate a short polynomial, and still conform. The FFT's
 * twiddle factors are `cos(angle)` and `sin(angle)` with `angle` running over
 * the whole of `[-2pi, 2pi)`, which is exactly where a weak argument reduction
 * shows up — so when the mixed-radix check reads 1.4e-6 on one adapter and
 * 3.3e-4 on another, this is the first thing to measure rather than the last.
 *
 * The probe computes the angle the same way the kernel does, and reports the
 * device's angle alongside its sine and cosine so the host can separate two
 * quite different faults:
 *
 *   * the **argument**: `TWO_PI * f32(num) / f32(den)` is rounded in f32, which
 *     costs about 6e-8 relative however good the transcendental is; and
 *   * the **transcendental**: `cos` of the angle the device itself computed,
 *     against an f64 cosine of that same angle. Nothing about the argument's
 *     own error enters this, so it is the implementation's quality alone.
 *
 * A device can be fine on one and bad on the other, and the fix differs: the
 * first wants the angle computed differently, the second wants `sin`/`cos`
 * avoided or their argument reduced before the call.
 */

import { dispatch, readBuffer, storageBuffer, uploadBuffer } from "./compute.js";
import { storageUsage, uniformUsage } from "./usage.js";

const TRIG_PROBE = /* wgsl */ `
struct Params { count: u32 };

@group(0) @binding(0) var<storage, read> numerator: array<i32>;
@group(0) @binding(1) var<storage, read> denominator: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_angle: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_cos: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_sin: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

// The same constant the FFT kernel uses, so the argument is bit-identical.
const TWO_PI: f32 = 6.283185307179586;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  // 🔴 Exactly \`fft.wgsl\`'s expression, including the order of operations:
  // multiply by TWO_PI first, then divide. Rewriting it here would measure a
  // different kernel.
  let angle = TWO_PI * f32(numerator[i]) / f32(denominator[i]);
  out_angle[i] = angle;
  out_cos[i] = cos(angle);
  out_sin[i] = sin(angle);
}
`;

export interface TrigProbe {
  /** The angle the device computed, f32. */
  readonly angle: Float32Array;
  readonly cosine: Float32Array;
  readonly sine: Float32Array;
}

/** Run `cos` and `sin` on the device for each `numerator[i] / denominator[i]`. */
export async function probeTrig(
  device: GPUDevice,
  numerator: Int32Array,
  denominator: Uint32Array,
): Promise<TrigProbe> {
  const count = numerator.length;
  if (denominator.length !== count) {
    throw new RangeError("numerator and denominator must be the same length");
  }
  const num = uploadBuffer(device, {
    label: "trig numerator",
    data: numerator,
    usage: storageUsage(),
  });
  const den = uploadBuffer(device, {
    label: "trig denominator",
    data: denominator,
    usage: storageUsage(),
  });
  const angle = storageBuffer(device, "trig angle", count * 4);
  const cosine = storageBuffer(device, "trig cos", count * 4);
  const sine = storageBuffer(device, "trig sin", count * 4);
  const params = uploadBuffer(device, {
    label: "trig params",
    data: new Uint32Array([count]),
    usage: uniformUsage(),
  });
  try {
    await dispatch(device, {
      label: "trig probe",
      code: TRIG_PROBE,
      bindings: [
        { binding: 0, resource: { buffer: num } },
        { binding: 1, resource: { buffer: den } },
        { binding: 2, resource: { buffer: angle } },
        { binding: 3, resource: { buffer: cosine } },
        { binding: 4, resource: { buffer: sine } },
        { binding: 5, resource: { buffer: params } },
      ],
      workgroups: [Math.ceil(count / 64)],
    });
    return {
      angle: new Float32Array(await readBuffer(device, angle, count * 4)),
      cosine: new Float32Array(await readBuffer(device, cosine, count * 4)),
      sine: new Float32Array(await readBuffer(device, sine, count * 4)),
    };
  } finally {
    for (const b of [num, den, angle, cosine, sine, params]) b.destroy();
  }
}

/**
 * The denominators the FFT actually asks for, and every numerator it can pass.
 *
 * `twiddle_index * r` is strictly below `ns * radix` in `fft.wgsl`, so the
 * angle stays inside one turn; the sign is the transform's direction, which is
 * -1 for a forward transform, so negative arguments are included rather than
 * assumed symmetric.
 */
export function fftTwiddleArguments(): {
  numerator: Int32Array;
  denominator: Uint32Array;
} {
  // The grids the FFT check transforms, factored: radices 2, 3, 5 and 7, the
  // products that appear as `ns * radix`, and 54's mixed stages.
  const denominators = [2, 3, 4, 5, 6, 7, 8, 9, 25, 27, 49, 54];
  const numerator: number[] = [];
  const denominator: number[] = [];
  for (const den of denominators) {
    for (let num = -(den - 1); num <= den - 1; num++) {
      numerator.push(num);
      denominator.push(den);
    }
  }
  return {
    numerator: Int32Array.from(numerator),
    denominator: Uint32Array.from(denominator),
  };
}
