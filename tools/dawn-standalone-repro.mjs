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

const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (adapter === null) {
  say("no adapter; nothing to reproduce");
  process.exit(0);
}
say(`adapter: ${JSON.stringify(adapter.info ?? null)}`);

const device = await adapter.requestDevice();
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
