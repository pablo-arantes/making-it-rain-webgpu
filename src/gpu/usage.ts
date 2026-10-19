/**
 * Buffer usage flags, read lazily.
 *
 * `GPUBufferUsage` is a global that exists only where WebGPU does, so a
 * module-level `const STORAGE = GPUBufferUsage.STORAGE | ...` throws on import
 * in Node -- not when a function runs, but when the module loads. That takes
 * out the no-GPU lane, which is the one `npm run check` and CI depend on, and
 * it does it from whichever file happens to sit on the import path. The failure
 * names the constant rather than the import that triggered it, so it reads like
 * a missing device instead of a module that cannot be loaded at all.
 *
 * Functions instead of constants: the flags are read when a buffer is actually
 * being made, by which point a device exists. test/module-load.test.ts imports
 * every module under src/gpu in the no-GPU lane to keep it that way.
 */

/** Read and written by kernels, uploadable from the host. */
export const storageUsage = (): GPUBufferUsageFlags =>
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

/** The same, and readable back to the host. */
export const readableStorageUsage = (): GPUBufferUsageFlags =>
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** A uniform block the host writes. */
export const uniformUsage = (): GPUBufferUsageFlags =>
  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
