/**
 * The one place GPU memory is allocated.
 *
 * `AGENTS.md` requires it: route every allocation through a single owner so
 * peak memory stays measurable. Until now each force evaluation created its
 * buffers and destroyed them, which is fine for a single-point energy and
 * useless for a step loop -- a 26,749-atom system would churn about 30 MB per
 * step through the driver's allocator.
 *
 * Two kinds of buffer, deliberately separated:
 *
 *   * **named** buffers persist for the life of the pool. Positions, velocities,
 *     forces, the cell list, the PME grids: everything a step reads or writes.
 *     Asking for the same name twice returns the same buffer, and asking for it
 *     at a different size is an error rather than a silent reallocation.
 *   * **scratch** buffers are leased and returned. A kernel that needs a
 *     temporary gets one at least as large as it asked for, and the pool reuses
 *     it afterwards.
 *
 * WebGPU gives no way to query free video memory, so `peakBytes` is what this
 * pool handed out, not what the device holds. That is still the number worth
 * reporting: it is the one we control.
 */

export interface BufferRequest {
  readonly label: string;
  readonly byteLength: number;
  readonly usage: GPUBufferUsageFlags;
}

export class AllocationMismatch extends Error {
  constructor(label: string, had: number, wanted: number) {
    super(
      `buffer "${label}" already exists at ${had} bytes and was requested at ` +
        `${wanted}. A named buffer's size is fixed for the life of the pool, so ` +
        "that a step loop cannot silently reallocate underneath itself. Use a " +
        "scratch lease for anything whose size varies.",
    );
    this.name = "AllocationMismatch";
  }
}

export class PoolDestroyed extends Error {
  constructor(what: string) {
    super(`the buffer pool has been destroyed; cannot ${what}`);
    this.name = "PoolDestroyed";
  }
}

interface Entry {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly usage: GPUBufferUsageFlags;
}

export interface PoolReport {
  readonly namedCount: number;
  readonly namedBytes: number;
  readonly scratchCount: number;
  readonly scratchBytes: number;
  readonly totalBytes: number;
  readonly peakBytes: number;
  readonly outstandingLeases: number;
}

/** WebGPU requires buffer sizes to be a multiple of four. */
const aligned = (bytes: number): number => Math.max(4, Math.ceil(bytes / 4) * 4);

export class BufferPool {
  private readonly named = new Map<string, Entry>();
  private readonly free: Entry[] = [];
  private readonly leased = new Set<GPUBuffer>();
  private liveBytes = 0;
  private peak = 0;
  private destroyed = false;

  constructor(readonly device: GPUDevice) {}

  /**
   * A buffer that lives as long as the pool.
   *
   * Idempotent by label: the second call returns the first buffer, which is what
   * lets a step loop ask for "positions" without threading it through every
   * function that needs it.
   */
  buffer(request: BufferRequest): GPUBuffer {
    if (this.destroyed) throw new PoolDestroyed(`allocate "${request.label}"`);
    const size = aligned(request.byteLength);
    const existing = this.named.get(request.label);
    if (existing !== undefined) {
      if (existing.byteLength !== size) {
        throw new AllocationMismatch(request.label, existing.byteLength, size);
      }
      return existing.buffer;
    }
    const buffer = this.device.createBuffer({
      label: request.label,
      size,
      usage: request.usage,
    });
    this.named.set(request.label, { buffer, byteLength: size, usage: request.usage });
    this.liveBytes += size;
    this.peak = Math.max(this.peak, this.liveBytes);
    return buffer;
  }

  /** A named buffer that already exists, or undefined. */
  existing(label: string): GPUBuffer | undefined {
    return this.named.get(label)?.buffer;
  }

  /**
   * Borrow a temporary.
   *
   * Reuses a free buffer of at least the requested size with compatible usage,
   * so a kernel run every step does not allocate every step. The returned
   * buffer may be larger than asked for; a bind group should name the range it
   * means rather than assume the whole buffer.
   */
  lease(request: BufferRequest): GPUBuffer {
    if (this.destroyed) throw new PoolDestroyed(`lease "${request.label}"`);
    const size = aligned(request.byteLength);
    // Best fit, so a large free buffer is not consumed by a small request.
    let bestIndex = -1;
    for (let i = 0; i < this.free.length; i++) {
      const candidate = this.free[i] as Entry;
      if (candidate.byteLength < size) continue;
      if ((candidate.usage & request.usage) !== request.usage) continue;
      if (
        bestIndex < 0 ||
        candidate.byteLength < (this.free[bestIndex] as Entry).byteLength
      ) {
        bestIndex = i;
      }
    }
    if (bestIndex >= 0) {
      const [entry] = this.free.splice(bestIndex, 1) as [Entry];
      this.leased.add(entry.buffer);
      return entry.buffer;
    }
    const buffer = this.device.createBuffer({
      label: `${request.label} (scratch)`,
      size,
      usage: request.usage,
    });
    this.leased.add(buffer);
    this.liveBytes += size;
    this.peak = Math.max(this.peak, this.liveBytes);
    return buffer;
  }

  /** Give a leased buffer back. */
  release(buffer: GPUBuffer): void {
    if (!this.leased.delete(buffer)) {
      throw new Error(
        `buffer "${buffer.label}" was released without being leased, which ` +
          "means either a double release or a named buffer being treated as " +
          "scratch.",
      );
    }
    this.free.push({
      buffer,
      byteLength: buffer.size,
      usage: buffer.usage,
    });
  }

  /** Run `use` with a lease, returning it even if `use` throws. */
  async withLease<T>(
    request: BufferRequest,
    use: (buffer: GPUBuffer) => Promise<T>,
  ): Promise<T> {
    const buffer = this.lease(request);
    try {
      return await use(buffer);
    } finally {
      this.release(buffer);
    }
  }

  report(): PoolReport {
    let namedBytes = 0;
    for (const entry of this.named.values()) namedBytes += entry.byteLength;
    let scratchBytes = 0;
    for (const entry of this.free) scratchBytes += entry.byteLength;
    for (const buffer of this.leased) scratchBytes += buffer.size;
    return {
      namedCount: this.named.size,
      namedBytes,
      scratchCount: this.free.length + this.leased.size,
      scratchBytes,
      totalBytes: namedBytes + scratchBytes,
      peakBytes: this.peak,
      outstandingLeases: this.leased.size,
    };
  }

  /**
   * Release everything.
   *
   * An outstanding lease at this point is a leak in the caller, not here, so it
   * is reported rather than quietly cleaned up.
   */
  destroy(): { readonly leaked: number } {
    const leaked = this.leased.size;
    for (const entry of this.named.values()) entry.buffer.destroy();
    for (const entry of this.free) entry.buffer.destroy();
    for (const buffer of this.leased) buffer.destroy();
    this.named.clear();
    this.free.length = 0;
    this.leased.clear();
    this.liveBytes = 0;
    this.destroyed = true;
    return { leaked };
  }
}
