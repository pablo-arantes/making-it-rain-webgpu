/**
 * Getting a WebGPU device in Node, through Dawn.
 *
 * This lane is opt-in (`MIR_GPU_TESTS=1`) and reports precisely why it is
 * unavailable when it is, because "skipped" and "broken" look the same
 * otherwise.  On macOS 14 the published `webgpu` binaries do not load at all --
 * every version from 0.3.10 to 0.6.0 is built against macOS 15 or 26 and wants
 * `MTLLogStateDescriptor` -- so on this machine the browser lane
 * (`npm run test:browser`) is the one that exercises the kernels.  See
 * CLAUDE.md.
 */

export interface DawnDevice {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
}

export type DawnStatus =
  | { readonly kind: "ready"; readonly handles: DawnDevice }
  | { readonly kind: "opted-out" }
  | { readonly kind: "unavailable"; readonly reason: string };

let cached: DawnStatus | undefined;

export async function dawn(): Promise<DawnStatus> {
  if (cached !== undefined) return cached;
  if (process.env.MIR_GPU_TESTS !== "1") {
    cached = { kind: "opted-out" };
    return cached;
  }
  try {
    // Imported dynamically so the module never loads in the default lane.
    const webgpu = await import("webgpu");
    const { create, globals } = webgpu as unknown as {
      create: (flags: string[]) => GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    // `allow_unsafe_apis` is what gates the experimental subgroup-matrix
    // extension on every vendor; without it a native run measures an adapter
    // that looks less capable than the browser's.
    const gpu = create(["enable-dawn-features=allow_unsafe_apis"]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) {
      cached = { kind: "unavailable", reason: "Dawn loaded but returned no adapter" };
      return cached;
    }
    const { requestEngineDevice } = await import("../src/gpu/device.js");
    cached = { kind: "ready", handles: { adapter, device: await requestEngineDevice(adapter) } };
    return cached;
  } catch (error) {
    cached = {
      kind: "unavailable",
      reason: `the webgpu package would not load: ${String(error).split("\n")[0]}`,
    };
    return cached;
  }
}
