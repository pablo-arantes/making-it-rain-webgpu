/**
 * Every module under src/gpu loads without a GPU.
 *
 * Not a test of behaviour: a test that the no-GPU lane can import the code at
 * all. `GPUBufferUsage` and `GPUShaderStage` are globals that exist only where
 * WebGPU does, so a module-level `const STORAGE = GPUBufferUsage.STORAGE | ...`
 * throws when the *module loads* rather than when a buffer is made. That takes
 * out `npm run check` and CI, from whichever file happens to be on the import
 * path, and the error names the constant rather than the import that triggered
 * it -- so it reads like a missing device instead of an unloadable module.
 *
 * This went in after exactly that: a usage constant added to src/gpu/forces.ts
 * broke two unrelated test files, because `checks.ts` imports `pme.ts` imports
 * `forces.ts`. src/gpu/usage.ts holds the flags as functions now, and this
 * keeps them that way.
 */

import { readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";

const modules = readdirSync("src/gpu")
  .filter((name) => name.endsWith(".ts"))
  .sort();

describe("src/gpu modules", () => {
  it("has modules to check", () => {
    expect(modules.length).toBeGreaterThan(10);
  });

  it.each(modules)("imports %s without a GPU", async (name) => {
    expect(typeof globalThis.GPUBufferUsage).toBe("undefined");
    await expect(import(`../src/gpu/${name}`)).resolves.toBeTruthy();
  });
});
