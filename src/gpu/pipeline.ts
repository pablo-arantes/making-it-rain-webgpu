/**
 * Compiled pipelines, kept, and passes recorded instead of dispatched.
 *
 * `dispatch` in src/gpu/compute.ts does the whole job each time it is called:
 * compile the module, check its messages, derive a layout, build a bind group,
 * encode one pass, submit, and wait for the queue to drain. That is the right
 * shape for a single-point energy, where the compile is a one-off and the wait
 * is the result. It is the wrong shape for a step loop, where the same dozen
 * kernels run every step and the wait is a pipeline drain nobody asked for.
 *
 * Two things change here.
 *
 * **Pipelines are cached**, keyed by the source and entry point. Compiling WGSL
 * is the expensive part -- a step touches a dozen entry points across six
 * modules -- and the result is immutable, so there is no reason to do it twice.
 * `dispatch` uses this too, which is why the validation lanes got faster rather
 * than slower when it landed.
 *
 * **Passes are recorded, not submitted.** A `Pass` is a pipeline, a bind group
 * and a workgroup count: everything needed to encode, and nothing that needs
 * the queue. A step is a list of them, built once, and encoding a hundred steps
 * into one submission is then a loop over that list. The error scope moves out
 * to the submission, which is the only place it can live once the dispatches
 * stop being individually awaited.
 */

export interface Pass {
  readonly label: string;
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
  readonly workgroups: readonly [number, number?, number?];
}

export interface PipelineSpec {
  readonly label: string;
  readonly code: string;
  readonly entryPoint: string;
  /**
   * An explicit layout, for a module whose entry points share one buffer set.
   *
   * Without it `layout: "auto"` derives a layout per entry point from what that
   * one reaches, so a bind group built for the whole set is rejected. See
   * src/gpu/integrate.ts, where five phases share seven buffers.
   */
  readonly bindGroupLayout?: GPUBindGroupLayout;
}

/** Stable identities for layouts, so a pipeline key can mention one. */
const layoutIds = new WeakMap<GPUBindGroupLayout, number>();
let nextLayoutId = 1;

function layoutId(layout: GPUBindGroupLayout | undefined): string {
  if (layout === undefined) return "auto";
  const existing = layoutIds.get(layout);
  if (existing !== undefined) return String(existing);
  const id = nextLayoutId++;
  layoutIds.set(layout, id);
  return String(id);
}

export class PipelineCache {
  private readonly pipelines = new Map<string, Promise<GPUComputePipeline>>();

  constructor(readonly device: GPUDevice) {}

  /**
   * The pipeline for one entry point, compiled at most once.
   *
   * Compilation messages are read and errors thrown here rather than left to
   * surface as a console warning and a silently wrong result -- the one failure
   * a differential test cannot tell from a bad kernel (CLAUDE.md). Caching the
   * *promise* rather than the pipeline means two concurrent callers for the same
   * kernel share one compile instead of racing.
   */
  pipeline(spec: PipelineSpec): Promise<GPUComputePipeline> {
    // The source is part of the key: `composeKernel` prepends shared WGSL, so
    // two kernels with the same entry-point name are still different programs.
    const key = `${layoutId(spec.bindGroupLayout)}\u0000${spec.entryPoint}\u0000${spec.code}`;
    const cached = this.pipelines.get(key);
    if (cached !== undefined) return cached;
    const created = this.compile(spec);
    this.pipelines.set(key, created);
    return created;
  }

  private async compile(spec: PipelineSpec): Promise<GPUComputePipeline> {
    const module = this.device.createShaderModule({ label: spec.label, code: spec.code });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === "error");
    if (errors.length > 0) {
      throw new Error(
        `${spec.label} failed to compile:\n` +
          errors.map((m) => `  ${m.lineNum}:${m.linePos} ${m.message}`).join("\n"),
      );
    }
    return this.device.createComputePipeline({
      label: spec.label,
      layout:
        spec.bindGroupLayout === undefined
          ? "auto"
          : this.device.createPipelineLayout({
              label: spec.label,
              bindGroupLayouts: [spec.bindGroupLayout],
            }),
      compute: { module, entryPoint: spec.entryPoint },
    });
  }

  /** A pass ready to encode. */
  async pass(
    spec: PipelineSpec,
    bindings: readonly { binding: number; resource: GPUBindingResource }[],
    workgroups: readonly [number, number?, number?],
  ): Promise<Pass> {
    const pipeline = await this.pipeline(spec);
    const bindGroup = this.device.createBindGroup({
      label: spec.label,
      layout: spec.bindGroupLayout ?? pipeline.getBindGroupLayout(0),
      entries: bindings.map(({ binding, resource }) => ({ binding, resource })),
    });
    return { label: spec.label, pipeline, bindGroup, workgroups };
  }

  get size(): number {
    return this.pipelines.size;
  }
}

/**
 * Encode passes into an existing encoder.
 *
 * One compute pass per `Pass`, because consecutive dispatches in a single
 * WebGPU compute pass have no ordering guarantee between them: the force
 * kernels must finish before the integrator reads their output, and separate
 * passes are what expresses that. WebGPU has no intra-pass barrier.
 */
export function encodePasses(
  encoder: GPUCommandEncoder,
  passes: readonly Pass[],
): void {
  for (const item of passes) {
    const pass = encoder.beginComputePass({ label: item.label });
    pass.setPipeline(item.pipeline);
    pass.setBindGroup(0, item.bindGroup);
    pass.dispatchWorkgroups(
      item.workgroups[0],
      item.workgroups[1] ?? 1,
      item.workgroups[2] ?? 1,
    );
    pass.end();
  }
}

/**
 * Submit passes and wait, with a validation scope around the lot.
 *
 * For a single evaluation. A loop encodes many steps and submits once, with its
 * own scope; see src/gpu/loop.ts.
 */
export async function submitPasses(
  device: GPUDevice,
  label: string,
  passes: readonly Pass[],
): Promise<void> {
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder({ label });
  encodePasses(encoder, passes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const error = await device.popErrorScope();
  if (error !== null) throw new Error(`${label}: ${error.message}`);
}
