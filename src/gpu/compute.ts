/**
 * The smallest useful compute helper: make a pipeline, bind buffers, dispatch,
 * read back.
 *
 * This is the single-evaluation path -- a force at one set of coordinates, a
 * probe, a validation run -- where the compile is a one-off and the wait for
 * the queue *is* the result. A step loop needs the opposite shape and uses
 * src/gpu/pipeline.ts: pipelines kept, passes recorded, one submission for many
 * steps. `dispatch` shares that module's pipeline cache, so a kernel invoked
 * once per fixture here is compiled once per device rather than once per call.
 *
 * Buffers here still have explicit lifetimes rather than coming from the pool.
 * That is deliberate for a one-shot evaluation: nothing outlives the call, so
 * there is nothing for a pool to amortise, and `AGENTS.md`'s requirement is
 * about the memory a *run* holds.
 */

import { PipelineCache, submitPasses } from "./pipeline.js";

export interface BufferSpec {
  readonly label: string;
  /** Initial contents.  Its byteLength sets the buffer size. */
  readonly data: ArrayBufferView;
  readonly usage: GPUBufferUsageFlags;
}

export function uploadBuffer(device: GPUDevice, spec: BufferSpec): GPUBuffer {
  const buffer = device.createBuffer({
    label: spec.label,
    // WebGPU requires a multiple of 4; every array we use is already aligned,
    // but rounding here keeps a caller from having to care.
    size: Math.ceil(spec.data.byteLength / 4) * 4,
    usage: spec.usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, spec.data as unknown as BufferSource);
  return buffer;
}

export function storageBuffer(
  device: GPUDevice,
  label: string,
  byteLength: number,
): GPUBuffer {
  return device.createBuffer({
    label,
    size: Math.ceil(byteLength / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

/** Copy a storage buffer back to the host. */
export async function readBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  byteLength: number,
): Promise<ArrayBuffer> {
  const size = Math.ceil(byteLength / 4) * 4;
  const staging = device.createBuffer({
    label: `readback(${source.label})`,
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: "readback" });
  encoder.copyBufferToBuffer(source, 0, staging, 0, size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const copy = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return copy;
}

/**
 * Run a single compute pass.
 *
 * `errorScope` is on by default: a WGSL validation error otherwise surfaces as
 * a console warning and a silently wrong result, which is the one failure mode
 * a differential test cannot distinguish from a bad kernel.
 */
const caches = new WeakMap<GPUDevice, PipelineCache>();

/** One pipeline cache per device, shared with the loop. */
export function pipelineCacheFor(device: GPUDevice): PipelineCache {
  const existing = caches.get(device);
  if (existing !== undefined) return existing;
  const created = new PipelineCache(device);
  caches.set(device, created);
  return created;
}

export async function dispatch(
  device: GPUDevice,
  options: {
    readonly label: string;
    readonly code: string;
    readonly entryPoint?: string;
    /**
     * Explicit binding indices, not a positional list.
     *
     * `layout: "auto"` derives a layout from the bindings an entry point
     * actually uses, so two kernels in one module -- or one kernel that leaves
     * an index unused -- do not share a layout. Naming the index here lets each
     * kernel declare whatever set it needs without the host having to pad gaps.
     */
    readonly bindings: readonly { binding: number; resource: GPUBindingResource }[];
    readonly workgroups: readonly [number, number?, number?];
    /**
     * An explicit layout, for a module whose entry points share one buffer set.
     *
     * `layout: "auto"` derives the layout from what *one* entry point reaches,
     * so a module where five of them read overlapping subsets -- the
     * integrator -- gets five incompatible layouts and rejects any bind group
     * built for the whole set. Declaring the layout makes it the host's, and
     * then a binding an entry point happens not to use is allowed to be there.
     */
    readonly bindGroupLayout?: GPUBindGroupLayout;
  },
): Promise<void> {
  const pass = await pipelineCacheFor(device).pass(
    {
      label: options.label,
      code: options.code,
      entryPoint: options.entryPoint ?? "main",
      ...(options.bindGroupLayout === undefined
        ? {}
        : { bindGroupLayout: options.bindGroupLayout }),
    },
    options.bindings,
    options.workgroups,
  );
  await submitPasses(device, options.label, [pass]);
}
