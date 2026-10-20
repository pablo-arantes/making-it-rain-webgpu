/**
 * The DCD writer against OpenMM's, byte for byte.
 *
 * "A reader accepted it" is not a test for this format. Every common way of
 * getting DCD wrong produces a file that opens: nanometres instead of ångström
 * reads as a system ten times too small, a raw timestep makes every time axis
 * wrong by a constant, and the box record's interleaved lengths and cosines
 * will happily be read in the wrong order. So the gate is equality with a
 * writer known to be right.
 *
 * The two 80-character title lines are the only part that cannot match: OpenMM
 * writes the current time into the second one. Everything else -- header,
 * record markers, unit cells, coordinates -- is compared exactly.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  AKMA_PICOSECONDS,
  DCD_HEADER_BYTES,
  DcdFormatError,
  DcdWriter,
  readDcd,
} from "../src/analysis/dcd.js";
import { readNpy } from "./npy.js";
import type { Vec3 } from "../src/engine/system.js";

interface DcdFacts {
  readonly atoms: number;
  readonly frames: number;
  readonly timestep_ps: number;
  readonly interval: number;
  readonly first_step: number;
  readonly bytes: number;
  readonly title_offset: number;
  readonly title_bytes: number;
}

const dir = "fixtures/dcd";
const present = existsSync(`${dir}/facts.json`);

describe.skipIf(!present)("DcdWriter", () => {
  const facts = present
    ? (JSON.parse(readFileSync(`${dir}/facts.json`, "utf8")) as DcdFacts)
    : undefined;

  /** The same file this writer would produce for the fixture's data. */
  const write = (): { ours: Uint8Array; writer: DcdWriter } => {
    const positions = readNpy(`${dir}/positions.npy`).data;
    const boxes = readNpy(`${dir}/boxes.npy`).data;
    const stride = facts!.atoms * 3;
    const writer = new DcdWriter({
      particleCount: facts!.atoms,
      timeStep: facts!.timestep_ps,
      firstStep: facts!.first_step,
      interval: facts!.interval,
      periodic: true,
      // OpenMM's first line verbatim, so only its timestamped second line
      // differs and the skipped range is as small as possible.
      titles: ["Created by OpenMM", ""],
    });
    const chunks: Uint8Array[] = [writer.header()];
    for (let frame = 0; frame < facts!.frames; frame++) {
      chunks.push(
        writer.frame(
          positions.slice(frame * stride, (frame + 1) * stride),
          boxes.slice(frame * 3, frame * 3 + 3) as unknown as Vec3,
        ),
      );
    }
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const ours = new Uint8Array(total);
    let at = 0;
    for (const chunk of chunks) {
      ours.set(chunk, at);
      at += chunk.length;
    }
    writer.patch(ours);
    return { ours, writer };
  };

  it("produces a file the same size as OpenMM's", () => {
    const { ours } = write();
    expect(ours.length).toBe(facts!.bytes);
  });

  it("matches OpenMM byte for byte, outside the timestamped title", () => {
    const { ours } = write();
    const theirs = new Uint8Array(readFileSync(`${dir}/reference.dcd`));
    const titleStart = facts!.title_offset;
    const titleEnd = titleStart + facts!.title_bytes;
    const differing: number[] = [];
    for (let i = 0; i < theirs.length; i++) {
      if (i >= titleStart && i < titleEnd) continue;
      if (ours[i] !== theirs[i]) differing.push(i);
    }
    // Reported as offsets, because *where* a DCD differs says what is wrong:
    // under 196 is the header, a multiple of the frame stride is a record
    // marker, anything else is a coordinate.
    expect(
      differing.slice(0, 16),
      `${differing.length} bytes differ, first at ${differing[0]}`,
    ).toEqual([]);
  });

  it("keeps the header and the frame stride exact", () => {
    const { writer } = write();
    expect(writer.header().length).toBe(DCD_HEADER_BYTES);
    // Three coordinate records of `atoms` floats with a marker each side, plus
    // the 56-byte unit cell record.
    expect(writer.frameBytes).toBe(3 * (8 + facts!.atoms * 4) + 56);
    expect(DCD_HEADER_BYTES + facts!.frames * writer.frameBytes).toBe(facts!.bytes);
  });

  it("writes the timestep in AKMA units", () => {
    const { ours } = write();
    const delta = new DataView(ours.buffer).getFloat32(44, true);
    expect(delta).toBeCloseTo(facts!.timestep_ps / AKMA_PICOSECONDS, 4);
    // And not in picoseconds, which is the mistake this guards: 0.002 would be
    // a plausible-looking value in the same field.
    expect(delta).not.toBeCloseTo(facts!.timestep_ps, 4);
  });

  it("patches the frame count and the last step", () => {
    const { ours, writer } = write();
    const view = new DataView(ours.buffer);
    expect(writer.frameCount).toBe(facts!.frames);
    expect(view.getInt32(8, true)).toBe(facts!.frames);
    expect(view.getInt32(20, true)).toBe(
      facts!.first_step + (facts!.frames - 1) * facts!.interval,
    );
  });

  it("refuses a non-finite coordinate", () => {
    // OpenMM refuses too. A trajectory with a NaN in it reads as a valid file
    // and ruins every analysis downstream.
    const writer = new DcdWriter({ particleCount: 2, timeStep: 0.002 });
    const positions = new Float64Array(6);
    positions[4] = Number.NaN;
    expect(() => writer.frame(positions, [1, 1, 1] as unknown as Vec3)).toThrow(/NaN/);
    positions[4] = Number.POSITIVE_INFINITY;
    expect(() => writer.frame(positions, [1, 1, 1] as unknown as Vec3)).toThrow(
      /Infinity/,
    );
  });

  it("refuses a periodic frame with no box", () => {
    const writer = new DcdWriter({ particleCount: 2, timeStep: 0.002 });
    expect(() => writer.frame(new Float64Array(6))).toThrow(/needs a box/);
  });

  it("writes no unit cell when the system is not periodic", () => {
    const writer = new DcdWriter({
      particleCount: 2,
      timeStep: 0.002,
      periodic: false,
    });
    expect(writer.frameBytes).toBe(3 * (8 + 2 * 4));
    expect(new DataView(writer.header().buffer).getInt32(48, true)).toBe(0);
    expect(writer.frame(new Float64Array(6)).length).toBe(writer.frameBytes);
  });
});

describe("reading a DCD back", () => {
  /**
   * The writer existed first and that was enough while other people's tools
   * did the reading. The analyses changed it: Cα RMSD and the rest run over a
   * trajectory this code wrote.
   *
   * The round trip is the cheap test. The one with teeth is against a file
   * **OpenMM wrote**, because a writer and a reader wrong in the same way
   * round-trip perfectly — the same argument as for `System` XML.
   */
  it("round-trips what the writer produced", () => {
    const particleCount = 7;
    const writer = new DcdWriter({
      particleCount,
      timeStep: 0.002,
      interval: 250,
      firstStep: 1000,
      periodic: true,
    });
    const frames = 4;
    const positions: Float64Array[] = [];
    const boxes: Vec3[] = [];
    const parts: Uint8Array[] = [writer.header()];
    for (let frame = 0; frame < frames; frame++) {
      const p = new Float64Array(particleCount * 3);
      for (let i = 0; i < p.length; i++) p[i] = Math.sin(i + frame) * 2;
      const box: Vec3 = [3 + frame * 0.01, 4, 5];
      positions.push(p);
      boxes.push(box);
      parts.push(writer.frame(p, box));
    }
    const total = parts.reduce((n, part) => n + part.length, 0);
    const file = new Uint8Array(total);
    let at = 0;
    for (const part of parts) {
      file.set(part, at);
      at += part.length;
    }
    writer.patch(file);

    const read = readDcd(file);
    expect(read.particleCount).toBe(particleCount);
    expect(read.frameCount).toBe(frames);
    expect(read.firstStep).toBe(1000);
    expect(read.interval).toBe(250);
    expect(read.periodic).toBe(true);
    expect(read.timeStep).toBeCloseTo(0.002, 6);
    for (let frame = 0; frame < frames; frame++) {
      for (let i = 0; i < particleCount * 3; i++) {
        // f32 on disk, so this is the storage precision, not the reader's.
        expect(read.positions[frame * particleCount * 3 + i]).toBeCloseTo(
          positions[frame]![i]!,
          5,
        );
      }
      for (let axis = 0; axis < 3; axis++) {
        expect(read.boxes[frame * 3 + axis]).toBeCloseTo(boxes[frame]![axis]!, 6);
      }
    }
  });

  it("refuses a file it would otherwise misread", () => {
    // 🔴 These are the failures that matter. A DCD read with the wrong
    // endianness, or with the unit-cell flag misjudged, does not throw -- it
    // yields coordinates that are merely wrong, which ruins an analysis
    // silently. The Fortran record markers are the only thing that tells the
    // two apart, so they are checked rather than skipped.
    expect(() => readDcd(new Uint8Array(8))).toThrow(DcdFormatError);

    const header = new DcdWriter({
      particleCount: 3,
      timeStep: 0.002,
      interval: 1,
      firstStep: 0,
      periodic: false,
    }).header();

    const bigEndian = Uint8Array.from(header);
    new DataView(bigEndian.buffer).setInt32(0, 84, false);
    expect(() => readDcd(bigEndian)).toThrow(/big-endian/);

    const noMagic = Uint8Array.from(header);
    noMagic[4] = 0x58;
    expect(() => readDcd(noMagic)).toThrow(/CORD/);

    const xplor = Uint8Array.from(header);
    new DataView(xplor.buffer).setInt32(84, 0, true);
    expect(() => readDcd(xplor)).toThrow(/X-PLOR/);

    // A header with no frames after it.
    expect(() => readDcd(header)).toThrow(/no complete frame/);
  });
});
