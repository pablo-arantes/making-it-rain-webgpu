import { readFileSync, readdirSync } from "node:fs";
import { describe, expect, it } from "vitest";

/**
 * No kernel may use a WGSL reserved word as an identifier.
 *
 * WGSL reserves a long list of words it does not yet use, so that it can later.
 * Tint enforces only some of them today -- `reference` compiles, `from` and
 * `target` do not -- which makes this exactly the kind of thing that passes on
 * one backend and fails on the next. Two of these cost a round trip each
 * before the list went in here.
 *
 * Needs no GPU, so it runs in the lane that runs everywhere.
 */
const RESERVED = `NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await
become binding_array cast catch class co_await co_return co_yield coherent column_major common compile
compile_fragment concept const_cast consteval constexpr constinit crate debugger decltype delete demote
demote_to_helper do dynamic_cast enum explicit export extends extern external fallthrough filter final
finally friend from fxgroup get goto groupshared highp impl implements import inline instanceof interface
layout lowp macro macro_rules match mediump meta mod module move mutable namespace new nil noexcept noinline
nointerpolation non_coherent noncoherent noperspective null nullptr of operator package packoffset partition
pass patch pixelfragment precise precision premerge priv protected pub public readonly readwrite reference
regardless register reinterpret_cast require resource restrict self set shared sizeof smooth snorm static
static_assert static_cast std subroutine super target tempalte template this thread_local throw trait try
type typedef typeid typename typeof union unless unorm unsafe unsized use using varying virtual volatile
wgsl where with writeonly yield`
  .split(/\s+/)
  .filter((word) => word.length > 0);

describe("WGSL kernels", () => {
  const files = readdirSync("src/gpu/kernels").filter((f) => f.endsWith(".wgsl"));

  it("there are kernels to check", () => {
    expect(files.length).toBeGreaterThan(5);
  });

  it.each(files)("%s uses no reserved word as an identifier", (file) => {
    // Comments may say anything; code may not.
    const code = readFileSync(`src/gpu/kernels/${file}`, "utf8").replace(
      /\/\/.*$/gm,
      "",
    );
    const found: string[] = [];
    for (const word of RESERVED) {
      if (new RegExp(`(?<![A-Za-z0-9_])${word}(?![A-Za-z0-9_])`).test(code)) {
        found.push(word);
      }
    }
    expect(found).toEqual([]);
  });
});
