/**
 * The smallest useful compute helper: make a pipeline, bind buffers, dispatch,
 * read back.
 *
 * Phase 1 replaces the allocation here with a pooled allocator, because
 * `AGENTS.md` requires every GPU allocation to go through one place so peak
 * memory stays measurable.  Until there is something to pool, this is the
 * honest version: explicit buffers with explicit lifetimes.
 */

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
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ label: options.label, code: options.code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === "error");
  if (errors.length > 0) {
    throw new Error(
      `${options.label} failed to compile:\n` +
        errors.map((m) => `  ${m.lineNum}:${m.linePos} ${m.message}`).join("\n"),
    );
  }
  const pipeline = device.createComputePipeline({
    label: options.label,
    layout:
      options.bindGroupLayout === undefined
        ? "auto"
        : device.createPipelineLayout({
            label: options.label,
            bindGroupLayouts: [options.bindGroupLayout],
          }),
    compute: { module, entryPoint: options.entryPoint ?? "main" },
  });
  const group = device.createBindGroup({
    label: options.label,
    layout: options.bindGroupLayout ?? pipeline.getBindGroupLayout(0),
    entries: options.bindings.map(({ binding, resource }) => ({ binding, resource })),
  });
  const encoder = device.createCommandEncoder({ label: options.label });
  const pass = encoder.beginComputePass({ label: options.label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(
    options.workgroups[0],
    options.workgroups[1] ?? 1,
    options.workgroups[2] ?? 1,
  );
  pass.end();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const error = await device.popErrorScope();
  if (error !== null) throw new Error(`${options.label}: ${error.message}`);
}
