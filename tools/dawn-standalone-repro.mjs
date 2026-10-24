// A self-contained reproducer for a SIGSEGV in `webgpu` (dawn.node).
//
// Deliberately uses *nothing* from this repository: the `webgpu` package, the
// WebGPU API, a four-byte buffer and a three-line shader. Run it with
//
//     npm i webgpu && node dawn-standalone-repro.mjs
//
// Every line is written with `writeSync` because `console.log` buffers and a
// process that dies in native code never flushes, so the last line printed is
// the call that took it down.

import { writeSync } from "node:fs";

const say = (message) => writeSync(1, `${message}\n`);

const { create, globals } = await import("webgpu");
Object.assign(globalThis, globals);

// 🔴 Toggles, because the bare version of this script **passes** on both Apple
// Metal and llvmpipe while this project's own path segfaults. Three things
// differ, and one run each says which: the instance flag, the optional
// features, and the raised limits.
const FLAG = process.env.REPRO_FLAG === "1";
const FEATURES = process.env.REPRO_FEATURES === "1";
const LIMITS = process.env.REPRO_LIMITS === "1";
say(`config: flag=${FLAG} features=${FEATURES} limits=${LIMITS}`);

const gpu = create(FLAG ? ["enable-dawn-features=allow_unsafe_apis"] : []);
const adapter = await gpu.requestAdapter(
  FLAG ? { powerPreference: "high-performance" } : {},
);
if (adapter === null) {
  say("no adapter; nothing to reproduce");
  process.exit(0);
}
say(`adapter: ${JSON.stringify(adapter.info ?? null)}`);

const wanted = ["timestamp-query", "subgroups", "subgroup-size-control", "shader-f16"];
const requiredFeatures = FEATURES ? wanted.filter((f) => adapter.features.has(f)) : [];
const requiredLimits = {};
if (LIMITS) {
  // The shape `requestEngineDevice` uses: ask high, clamp to the adapter.
  for (const [key, ask] of [
    ["maxStorageBufferBindingSize", 2 ** 31],
    ["maxBufferSize", 2 ** 31],
    ["maxComputeInvocationsPerWorkgroup", 1024],
    ["maxComputeWorkgroupStorageSize", 32768],
  ]) {
    const available = adapter.limits[key];
    if (typeof available === "number") requiredLimits[key] = Math.min(ask, available);
  }
}
say(`features: [${requiredFeatures.join(", ")}]`);
say(`limits: ${JSON.stringify(requiredLimits)}`);
const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
say("device created");
device.lost?.then((r) => say(`device lost: ${r?.message ?? "?"}`)).catch(() => {});

const module = device.createShaderModule({
  code: `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main() { out[0] = out[0] + 1u; }
`,
});
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
say("pipeline created");

for (let pass = 1; pass <= 3; pass++) {
  say(`pass ${pass}: createBuffer`);
  const buffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer } }],
  });

  say(`pass ${pass}: encode + submit`);
  const encoder = device.createCommandEncoder();
  const compute = encoder.beginComputePass();
  compute.setPipeline(pipeline);
  compute.setBindGroup(0, bindGroup);
  compute.dispatchWorkgroups(1);
  compute.end();
  device.queue.submit([encoder.finish()]);

  say(`pass ${pass}: copy + mapAsync`);
  const staging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const copy = device.createCommandEncoder();
  copy.copyBufferToBuffer(buffer, 0, staging, 0, 4);
  device.queue.submit([copy.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const value = new Uint32Array(staging.getMappedRange().slice(0))[0];
  staging.unmap();
  staging.destroy();
  say(`pass ${pass}: out[0] = ${value}`);
}

say("completed all three passes");
