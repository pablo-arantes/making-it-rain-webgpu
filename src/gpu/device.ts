/**
 * Adapter and device acquisition, and what we tell the user about their GPU.
 *
 * Two jobs.  Getting a device with the limits this workload needs, and
 * describing the result honestly -- including an estimated maximum system
 * size, which is the question every user actually has.
 */

/** What the engine asks for beyond WebGPU's defaults. */
export interface DeviceRequirements {
  /** Largest single buffer, bytes.  The neighbour list is the big one. */
  readonly maxBufferSize: number;
  /** Largest storage binding, bytes. */
  readonly maxStorageBufferBindingSize: number;
  /** Workgroup storage for the FFT and reduction kernels, bytes. */
  readonly maxComputeWorkgroupStorageSize: number;
}

/** WebGPU's guaranteed floor, which is what an adapter gives if we ask for nothing. */
export const WEBGPU_DEFAULT_LIMITS: DeviceRequirements = {
  maxBufferSize: 256 * 1024 * 1024,
  maxStorageBufferBindingSize: 128 * 1024 * 1024,
  maxComputeWorkgroupStorageSize: 16384,
};

/**
 * Features we use when offered and work without.  Nothing here may change a
 * result: `timestamp-query` is measurement, `subgroups` and `shader-f16` select
 * a faster kernel that must still pass the same differential test.
 */
export const OPTIONAL_FEATURES = [
  "timestamp-query",
  "subgroups",
  "subgroup-size-control",
  "shader-f16",
] as const satisfies readonly GPUFeatureName[];

export interface AdapterReport {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly features: readonly string[];
  readonly limits: Readonly<Record<string, number>>;
  /** True when the adapter admits to being a software rasteriser. */
  readonly isFallback: boolean;
}

const REPORTED_LIMITS = [
  "maxBufferSize",
  "maxStorageBufferBindingSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupsPerDimension",
  "maxStorageBuffersPerShaderStage",
  "maxUniformBufferBindingSize",
] as const;

export class WebGpuUnavailable extends Error {
  constructor(readonly reason: "no-navigator-gpu" | "no-adapter" | "no-device") {
    super(
      {
        "no-navigator-gpu":
          "This browser does not expose WebGPU. Chrome or Edge 113+, or Safari 18+, " +
          "will work. WebGPU is also only available in a secure context, so an " +
          "http:// page that is not localhost will not see it.",
        "no-adapter":
          "WebGPU is present but no GPU adapter was returned. On Linux this is " +
          "usually a driver or Vulkan configuration problem.",
        "no-device":
          "An adapter was found but it would not give a device with the limits " +
          "this engine needs.",
      }[reason],
    );
    this.name = "WebGpuUnavailable";
  }
}

export async function requestEngineAdapter(): Promise<GPUAdapter> {
  if (typeof navigator === "undefined" || navigator.gpu === undefined) {
    throw new WebGpuUnavailable("no-navigator-gpu");
  }
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (adapter === null) throw new WebGpuUnavailable("no-adapter");
  return adapter;
}

/**
 * Request a device, asking for the limits we need but never more than the
 * adapter supports.
 *
 * Asking for a limit the adapter cannot meet fails the whole request, so each
 * one is clamped and the clamp is visible in the report rather than silent.
 */
export async function requestEngineDevice(
  adapter: GPUAdapter,
  requirements: Partial<DeviceRequirements> = {},
): Promise<GPUDevice> {
  const wanted: DeviceRequirements = { ...WEBGPU_DEFAULT_LIMITS, ...requirements };
  const limits: Record<string, number> = {};
  for (const [key, value] of Object.entries(wanted)) {
    const available = adapter.limits[key as keyof GPUSupportedLimits];
    if (typeof available === "number") limits[key] = Math.min(value, available);
  }
  const features = OPTIONAL_FEATURES.filter((f) => adapter.features.has(f));
  const device = await adapter.requestDevice({
    requiredLimits: limits,
    requiredFeatures: features,
  });
  if (device === undefined) throw new WebGpuUnavailable("no-device");
  return device;
}

export function describeAdapter(adapter: GPUAdapter): AdapterReport {
  const info: Partial<GPUAdapterInfo> = adapter.info ?? {};
  const limits: Record<string, number> = {};
  for (const key of REPORTED_LIMITS) {
    const value = adapter.limits[key];
    if (typeof value === "number") limits[key] = value;
  }
  return {
    vendor: info.vendor ?? "",
    architecture: info.architecture ?? "",
    device: info.device ?? "",
    description: info.description ?? "",
    features: [...adapter.features].sort(),
    limits,
    // `isFallbackAdapter` moved onto `info`; read both so either spelling works.
    isFallback:
      (info as { isFallbackAdapter?: boolean }).isFallbackAdapter ??
      (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter ??
      false,
  };
}

/**
 * Bytes of GPU memory per atom, by term.  From docs/PLAN.md section Q5; the
 * numbers are a model, not a measurement, and Phase 1 replaces them with one.
 */
export const BYTES_PER_ATOM = {
  /** Positions in double-single: 3 components x 2 words x 4 bytes. */
  positions: 24,
  /** Velocities in f32. */
  velocities: 12,
  /** Forces in 64-bit fixed point: 3 components x 2 words x 4 bytes. */
  forces: 24,
  /** Sorted copies, cell indices, constraint and exclusion bookkeeping. */
  bookkeeping: 48,
  /**
   * The full neighbour list, which dominates everything else.  At a 1.0 nm
   * cutoff with a 0.2 nm skin a sphere at water density holds about 240
   * neighbours, and a gather kernel needs the full list rather than half of
   * it, at 4 bytes an entry.
   */
  neighbours: 240 * 4,
} as const;

export const TOTAL_BYTES_PER_ATOM = Object.values(BYTES_PER_ATOM).reduce(
  (a, b) => a + b,
  0,
);

export interface SizeEstimate {
  readonly bytesPerAtom: number;
  /** Where the neighbour list stops fitting in one storage binding. */
  readonly atomsAtBindingLimit: number;
  /** Where the whole working set stops fitting in the largest buffer we can make. */
  readonly atomsAtBufferLimit: number;
  readonly maxAtoms: number;
  readonly limitedBy: "storage binding" | "buffer size";
}

/**
 * An estimated maximum system size for an adapter.
 *
 * WebGPU deliberately does not expose free VRAM, so this is a limits-derived
 * ceiling and not a promise about physical memory.  It is reported as an
 * estimate, and the run loop checks the real allocations before starting.
 */
export function estimateMaxAtoms(limits: GPUSupportedLimits): SizeEstimate {
  const binding = limits.maxStorageBufferBindingSize;
  const buffer = limits.maxBufferSize;
  const atomsAtBindingLimit = Math.floor(binding / BYTES_PER_ATOM.neighbours);
  const atomsAtBufferLimit = Math.floor(buffer / TOTAL_BYTES_PER_ATOM);
  const maxAtoms = Math.min(atomsAtBindingLimit, atomsAtBufferLimit);
  return {
    bytesPerAtom: TOTAL_BYTES_PER_ATOM,
    atomsAtBindingLimit,
    atomsAtBufferLimit,
    maxAtoms,
    limitedBy:
      atomsAtBindingLimit <= atomsAtBufferLimit ? "storage binding" : "buffer size",
  };
}

/**
 * Watch for a lost device.
 *
 * A lost device is not recoverable and not rare -- a driver reset, a long
 * dispatch hitting the watchdog, or the OS reclaiming the GPU will all do it.
 * The run loop's answer is to stop, keep the last checkpoint and offer resume,
 * so the only thing this helper does is make the event impossible to miss.
 */
export function onDeviceLost(
  device: GPUDevice,
  handler: (info: GPUDeviceLostInfo) => void,
): void {
  void device.lost.then(handler);
}
