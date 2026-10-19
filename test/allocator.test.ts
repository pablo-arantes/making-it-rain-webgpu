import { describe, expect, it } from "vitest";
import {
  AllocationMismatch,
  BufferPool,
  PoolDestroyed,
} from "../src/gpu/allocator.js";

/**
 * The pool is bookkeeping, so it is tested against a stub device rather than a
 * real one -- which means it is tested in the lane that runs everywhere, not
 * only where there is a GPU. What a real device adds is whether the buffers
 * work, and the kernels' own differential tests cover that.
 */
interface StubBuffer {
  label: string;
  size: number;
  usage: number;
  destroyed: boolean;
}

function stubDevice(): { device: GPUDevice; created: StubBuffer[] } {
  const created: StubBuffer[] = [];
  const device = {
    createBuffer(descriptor: GPUBufferDescriptor): GPUBuffer {
      const buffer: StubBuffer = {
        label: descriptor.label ?? "",
        size: descriptor.size,
        usage: descriptor.usage,
        destroyed: false,
      };
      created.push(buffer);
      return {
        ...buffer,
        destroy() {
          buffer.destroyed = true;
        },
      } as unknown as GPUBuffer;
    },
  } as unknown as GPUDevice;
  return { device, created };
}

const STORAGE = 0x80;
const UNIFORM = 0x40;

describe("named buffers", () => {
  it("returns the same buffer for the same label", () => {
    const { device, created } = stubDevice();
    const pool = new BufferPool(device);
    const a = pool.buffer({ label: "positions", byteLength: 1024, usage: STORAGE });
    const b = pool.buffer({ label: "positions", byteLength: 1024, usage: STORAGE });
    expect(a).toBe(b);
    expect(created).toHaveLength(1);
  });

  it("refuses the same label at a different size", () => {
    const { device } = stubDevice();
    const pool = new BufferPool(device);
    pool.buffer({ label: "positions", byteLength: 1024, usage: STORAGE });
    expect(() =>
      pool.buffer({ label: "positions", byteLength: 2048, usage: STORAGE }),
    ).toThrow(AllocationMismatch);
    // The message has to say why, because the fix is to use a lease.
    expect(() =>
      pool.buffer({ label: "positions", byteLength: 2048, usage: STORAGE }),
    ).toThrow(/scratch lease/);
  });

  it("rounds sizes up to a multiple of four", () => {
    const { device, created } = stubDevice();
    const pool = new BufferPool(device);
    pool.buffer({ label: "odd", byteLength: 13, usage: STORAGE });
    expect(created[0]?.size).toBe(16);
  });

  it("never returns a zero-sized buffer", () => {
    const { device, created } = stubDevice();
    const pool = new BufferPool(device);
    pool.buffer({ label: "empty", byteLength: 0, usage: STORAGE });
    expect(created[0]?.size).toBe(4);
  });
});

describe("scratch leases", () => {
  it("reuses a returned buffer rather than allocating again", () => {
    const { device, created } = stubDevice();
    const pool = new BufferPool(device);
    const first = pool.lease({ label: "grid", byteLength: 4096, usage: STORAGE });
    pool.release(first);
    const second = pool.lease({ label: "grid", byteLength: 4096, usage: STORAGE });
    expect(second).toBe(first);
    expect(created).toHaveLength(1);
  });

  it("will hand back a larger buffer, but not a smaller one", () => {
    const { device, created } = stubDevice();
    const pool = new BufferPool(device);
    const big = pool.lease({ label: "big", byteLength: 8192, usage: STORAGE });
    pool.release(big);
    const small = pool.lease({ label: "small", byteLength: 1024, usage: STORAGE });
    expect(small).toBe(big);
    pool.release(small);
    // Asking for more than the free buffer holds has to allocate.
    pool.lease({ label: "bigger", byteLength: 16384, usage: STORAGE });
    expect(created).toHaveLength(2);
  });

  it("takes the best fit, so a large buffer is not eaten by a small request", () => {
    const { device } = stubDevice();
    const pool = new BufferPool(device);
    const big = pool.lease({ label: "big", byteLength: 8192, usage: STORAGE });
    const small = pool.lease({ label: "small", byteLength: 1024, usage: STORAGE });
    pool.release(big);
    pool.release(small);
    const asked = pool.lease({ label: "ask", byteLength: 512, usage: STORAGE });
    expect(asked).toBe(small);
  });

  it("will not reuse a buffer whose usage does not cover the request", () => {
    const { device, created } = stubDevice();
    const pool = new BufferPool(device);
    pool.release(pool.lease({ label: "u", byteLength: 256, usage: UNIFORM }));
    pool.lease({ label: "s", byteLength: 256, usage: STORAGE });
    expect(created).toHaveLength(2);
  });

  it("refuses a double release", () => {
    const { device } = stubDevice();
    const pool = new BufferPool(device);
    const buffer = pool.lease({ label: "x", byteLength: 256, usage: STORAGE });
    pool.release(buffer);
    expect(() => pool.release(buffer)).toThrow(/without being leased/);
  });

  it("returns a lease even when the body throws", async () => {
    const { device, created } = stubDevice();
    const pool = new BufferPool(device);
    await expect(
      pool.withLease({ label: "x", byteLength: 256, usage: STORAGE }, async () => {
        throw new Error("kernel failed");
      }),
    ).rejects.toThrow("kernel failed");
    expect(pool.report().outstandingLeases).toBe(0);
    // And the returned buffer is reused rather than leaked.
    pool.lease({ label: "x", byteLength: 256, usage: STORAGE });
    expect(created).toHaveLength(1);
  });
});

describe("reporting", () => {
  it("tracks the peak, which is what reuse is meant to hold down", () => {
    const { device } = stubDevice();
    const pool = new BufferPool(device);
    pool.buffer({ label: "resident", byteLength: 1000, usage: STORAGE });
    const a = pool.lease({ label: "a", byteLength: 4000, usage: STORAGE });
    pool.release(a);
    const b = pool.lease({ label: "b", byteLength: 4000, usage: STORAGE });
    pool.release(b);
    // The second lease reused the first, so the peak never saw 9000.
    expect(pool.report().peakBytes).toBe(5000);
    expect(pool.report().totalBytes).toBe(5000);
  });

  it("reports an outstanding lease at destruction as the caller's leak", () => {
    const { device } = stubDevice();
    const pool = new BufferPool(device);
    pool.lease({ label: "forgotten", byteLength: 256, usage: STORAGE });
    expect(pool.destroy().leaked).toBe(1);
  });

  it("refuses to allocate after destruction", () => {
    const { device } = stubDevice();
    const pool = new BufferPool(device);
    pool.destroy();
    expect(() =>
      pool.buffer({ label: "late", byteLength: 256, usage: STORAGE }),
    ).toThrow(PoolDestroyed);
    expect(() =>
      pool.lease({ label: "late", byteLength: 256, usage: STORAGE }),
    ).toThrow(PoolDestroyed);
  });
});
