/**
 * Every kernel stays inside the bindings WebGPU *guarantees*.
 *
 * 🔴 This is a portability test that the machine it was written on cannot
 * perform. `maxStorageBuffersPerShaderStage` has a mandated floor of 8, and
 * Apple reports far more — so a kernel that bound nine would compile here,
 * pass every differential check here, and fail to create its pipeline on a
 * conforming Intel or NVIDIA device. The first sign would be a user's bug
 * report, and the engine has never run on a non-Apple GPU.
 *
 * It also keeps `MAX_STORAGE_BUFFERS_USED` honest. That constant is what
 * `deviceFitness` refuses a device on, so a kernel quietly growing past it
 * would make the preflight promise something that then fails, and a kernel
 * shrinking below it would make the preflight refuse a device that would have
 * worked.
 *
 * The analysis is static and deliberately simple: declarations, function
 * bodies by brace matching, and a call graph. It reads the *composed* sources
 * from `kernels.ts`, not the raw files, because what matters is what is handed
 * to `createShaderModule` — the shared prelude declares bindings 0 to 2 for
 * every force kernel, and a module that also declared its own binding 0 would
 * not be visible in the file alone.
 */

import { describe, expect, it } from "vitest";
import * as kernels from "../src/gpu/kernels.js";
import { MAX_STORAGE_BUFFERS_USED } from "../src/gpu/device.js";

/** WebGPU's mandated floors, which are what a conforming device may offer. */
const GUARANTEED = {
  storageBuffersPerShaderStage: 8,
  uniformBuffersPerShaderStage: 12,
} as const;

/** Comments first: an identifier named in one is not a use of it. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
}

interface Declaration {
  readonly binding: number;
  readonly kind: "storage" | "uniform";
}

function declarations(source: string): Map<string, Declaration> {
  const found = new Map<string, Declaration>();
  const pattern = /@group\(0\)\s*@binding\((\d+)\)\s*var\s*<([^>]*)>\s*(\w+)/g;
  for (const match of source.matchAll(pattern)) {
    const space = match[2] as string;
    found.set(match[3] as string, {
      binding: Number(match[1]),
      kind: space.includes("uniform") ? "uniform" : "storage",
    });
  }
  return found;
}

/** Each function's body, by matching braces from its signature. */
function bodies(source: string): {
  readonly byName: Map<string, string>;
  readonly entryPoints: readonly string[];
} {
  const byName = new Map<string, string>();
  const entryPoints: string[] = [];
  const pattern = /(@compute[^\n]*\n\s*)?fn\s+(\w+)\s*\(/g;
  for (const match of source.matchAll(pattern)) {
    const name = match[2] as string;
    const open = source.indexOf("{", (match.index ?? 0) + match[0].length - 1);
    if (open === -1) continue;
    let depth = 0;
    let end = open;
    for (; end < source.length; end++) {
      if (source[end] === "{") depth++;
      else if (source[end] === "}" && --depth === 0) break;
    }
    byName.set(name, source.slice(open, end + 1));
    if (match[1] !== undefined) entryPoints.push(name);
  }
  return { byName, entryPoints };
}

function referenced(name: string, source: string): boolean {
  return new RegExp(`(?<![\\w_])${name}(?![\\w_])`).test(source);
}

/** The bindings one entry point uses, following the calls it makes. */
function usedBindings(
  entry: string,
  byName: Map<string, string>,
  declared: Map<string, Declaration>,
): { storage: Set<number>; uniform: Set<number> } {
  const storage = new Set<number>();
  const uniform = new Set<number>();
  const seen = new Set<string>();
  const visit = (name: string): void => {
    if (seen.has(name)) return;
    seen.add(name);
    const body = byName.get(name);
    if (body === undefined) return;
    for (const [identifier, declaration] of declared) {
      if (!referenced(identifier, body)) continue;
      (declaration.kind === "storage" ? storage : uniform).add(declaration.binding);
    }
    for (const call of body.matchAll(/(?<![\w_])(\w+)\s*\(/g)) {
      const callee = call[1] as string;
      if (callee !== name && byName.has(callee)) visit(callee);
    }
  };
  visit(entry);
  return { storage, uniform };
}

/** Every composed kernel source `kernels.ts` exports. */
const modules = Object.entries(kernels).filter(
  (entry): entry is [string, string] =>
    typeof entry[1] === "string" && entry[1].includes("@compute"),
);

describe("kernel bindings", () => {
  it("exports composed sources to analyse", () => {
    expect(modules.length).toBeGreaterThan(10);
  });

  const counts: { name: string; storage: number; uniform: number }[] = [];
  for (const [name, source] of modules) {
    const clean = stripComments(source);
    const declared = declarations(clean);
    const { byName, entryPoints } = bodies(clean);
    it(`${name} declares at least one entry point`, () => {
      expect(entryPoints.length).toBeGreaterThan(0);
    });
    for (const entry of entryPoints) {
      const used = usedBindings(entry, byName, declared);
      counts.push({
        name: `${name}.${entry}`,
        storage: used.storage.size,
        uniform: used.uniform.size,
      });
      it(`${name}.${entry} fits the bindings WebGPU guarantees`, () => {
        expect(
          used.storage.size,
          `uses storage bindings ${[...used.storage].sort((a, b) => a - b).join(", ")}`,
        ).toBeLessThanOrEqual(GUARANTEED.storageBuffersPerShaderStage);
        expect(used.uniform.size).toBeLessThanOrEqual(
          GUARANTEED.uniformBuffersPerShaderStage,
        );
      });
    }
  }

  it("agrees with the MAX_STORAGE_BUFFERS_USED that deviceFitness refuses on", () => {
    const worst = counts.reduce((a, b) => (b.storage > a.storage ? b : a));
    // Not an equality against the maximum found: the constant is allowed to
    // claim headroom over what is used, because asking a device for one more
    // than needed is safe while asking for one fewer is not. What it may not do
    // is claim *less* than a kernel uses, which is the direction that turns a
    // passing preflight into a failed pipeline.
    expect(
      worst.storage,
      `${worst.name} uses ${worst.storage} storage buffers`,
    ).toBeLessThanOrEqual(MAX_STORAGE_BUFFERS_USED);
    expect(MAX_STORAGE_BUFFERS_USED).toBeLessThanOrEqual(
      GUARANTEED.storageBuffersPerShaderStage,
    );
  });
});
