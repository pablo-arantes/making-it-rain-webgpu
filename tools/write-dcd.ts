/**
 * Write a DCD from the fixture's data, so a Python reader can be pointed at
 * *our* bytes rather than at OpenMM's.
 *
 * test/dcd.test.ts already proves the two files are byte-identical outside the
 * timestamped title, which makes "does it open" a question about the reference
 * file. Answering it on our own output anyway costs nothing and removes the
 * inference from the chain.
 *
 *   npx tsx tools/write-dcd.ts fixtures/dcd
 */

import { readFileSync, writeFileSync } from "node:fs";
import { DcdWriter } from "../src/analysis/dcd.js";
import { decodeNpy } from "../src/analysis/npy.js";

const dir = process.argv[2] ?? "fixtures/dcd";
const facts = JSON.parse(readFileSync(`${dir}/facts.json`, "utf8")) as {
  atoms: number;
  frames: number;
  timestep_ps: number;
  interval: number;
  first_step: number;
};

const load = (name: string): Float64Array => {
  const bytes = readFileSync(`${dir}/${name}`);
  return decodeNpy(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer,
    name,
  ).data;
};

const positions = load("positions.npy");
const boxes = load("boxes.npy");
const stride = facts.atoms * 3;
const writer = new DcdWriter({
  particleCount: facts.atoms,
  timeStep: facts.timestep_ps,
  firstStep: facts.first_step,
  interval: facts.interval,
  periodic: true,
  titles: ["Created by making-it-rain-webgpu", "fixture"],
});

const chunks: Uint8Array[] = [writer.header()];
for (let frame = 0; frame < facts.frames; frame++) {
  chunks.push(
    writer.frame(
      positions.slice(frame * stride, (frame + 1) * stride),
      [
        [boxes[frame * 3] as number, 0, 0],
        [0, boxes[frame * 3 + 1] as number, 0],
        [0, 0, boxes[frame * 3 + 2] as number],
      ],
    ),
  );
}
const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
const file = new Uint8Array(total);
let at = 0;
for (const chunk of chunks) {
  file.set(chunk, at);
  at += chunk.length;
}
writer.patch(file);
writeFileSync(`${dir}/ours.dcd`, file);
process.stdout.write(
  `wrote ${dir}/ours.dcd: ${total} bytes, ${writer.frameCount} frames of ${facts.atoms} atoms\n`,
);
