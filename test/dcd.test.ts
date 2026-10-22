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
  unitCellOf,
} from "../src/analysis/dcd.js";
import { readNpy } from "./npy.js";
import type { Vec3 } from "../src/engine/system.js";

interface DcdFacts {
  readonly shape?: string;
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

/** A rectangular cell from three edge lengths, which is most of these cases. */
const rectangular = (lengths: Vec3): readonly [Vec3, Vec3, Vec3] => [
  [lengths[0] as number, 0, 0],
  [0, lengths[1] as number, 0],
  [0, 0, lengths[2] as number],
];
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
          rectangular(boxes.slice(frame * 3, frame * 3 + 3) as unknown as Vec3),
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
    expect(() => writer.frame(positions, rectangular([1, 1, 1]))).toThrow(/NaN/);
    positions[4] = Number.POSITIVE_INFINITY;
    expect(() => writer.frame(positions, rectangular([1, 1, 1]))).toThrow(
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
      parts.push(writer.frame(p, rectangular(box)));
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

const triclinicDir = "fixtures/dcd-dodecahedron";
const triclinicPresent = existsSync(`${triclinicDir}/facts.json`);

/**
 * The same byte comparison, for a cell that is not rectangular.
 *
 * 🔴 This is the only test that can catch the unit cell being written as a
 * rectangle, and it was written because the writer *was* doing that. A DCD does
 * not store box vectors: it stores three edge lengths and three angle cosines,
 * `[a, cos gamma, b, cos beta, cos alpha, c]`. For a rhombic dodecahedron of
 * width `w` the three edges are all `w` and two of the angles are 60 degrees,
 * while the vectors' *diagonal* is `(w, w, 0.707w)` — so a writer handed the
 * diagonal and assuming right angles emits a cell with the correct volume, the
 * wrong shape, and nothing to signal it. For the 4 nm cell here OpenMM writes
 * `[40, 0, 40, 0.5, 0.5, 40]` and the old path wrote `[40, 0, 40, 0, 0, 28.28]`.
 *
 * A reader accepting the file proves nothing, which is why this is byte
 * equality against a DCD OpenMM wrote rather than a round trip.
 */
describe.skipIf(!triclinicPresent)("DcdWriter on a dodecahedral cell", () => {
  const facts = triclinicPresent
    ? (JSON.parse(readFileSync(`${triclinicDir}/facts.json`, "utf8")) as DcdFacts)
    : undefined;

  const write = () => {
    const positions = readNpy(`${triclinicDir}/positions.npy`).data;
    const vectors = readNpy(`${triclinicDir}/boxvectors.npy`).data;
    const stride = facts!.atoms * 3;
    const writer = new DcdWriter({
      particleCount: facts!.atoms,
      timeStep: facts!.timestep_ps,
      firstStep: facts!.first_step,
      interval: facts!.interval,
      periodic: true,
      titles: ["Created by OpenMM", ""],
    });
    const chunks: Uint8Array[] = [writer.header()];
    for (let frame = 0; frame < facts!.frames; frame++) {
      const cell: readonly [Vec3, Vec3, Vec3] = [
        [vectors[frame * 9]!, vectors[frame * 9 + 1]!, vectors[frame * 9 + 2]!],
        [vectors[frame * 9 + 3]!, vectors[frame * 9 + 4]!, vectors[frame * 9 + 5]!],
        [vectors[frame * 9 + 6]!, vectors[frame * 9 + 7]!, vectors[frame * 9 + 8]!],
      ];
      chunks.push(
        writer.frame(positions.slice(frame * stride, (frame + 1) * stride), cell),
      );
    }
    const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const ours = new Uint8Array(total);
    let at = 0;
    for (const chunk of chunks) {
      ours.set(chunk, at);
      at += chunk.length;
    }
    // The frame count and last step are patched in once the frames are known.
    writer.patch(ours);
    return { ours, writer };
  };

  it("is a dodecahedral fixture, with edges equal and two angles at 60°", () => {
    expect(facts!.shape).toBe("dodecahedron");
    const vectors = readNpy(`${triclinicDir}/boxvectors.npy`).data;
    const cell = unitCellOf([
      [vectors[0]!, vectors[1]!, vectors[2]!],
      [vectors[3]!, vectors[4]!, vectors[5]!],
      [vectors[6]!, vectors[7]!, vectors[8]!],
    ]);
    // All three edges the same, which is what makes it *rhombic* -- and what
    // the diagonal (w, w, 0.707w) does not show.
    expect(cell.lengths[0]).toBeCloseTo(cell.lengths[1]!, 12);
    expect(cell.lengths[1]).toBeCloseTo(cell.lengths[2]!, 12);
    expect(cell.cosines[0]).toBeCloseTo(0.5, 12);
    expect(cell.cosines[1]).toBeCloseTo(0.5, 12);
    expect(cell.cosines[2]).toBeCloseTo(0, 12);
    // The trap, stated as a number: the third edge and the third diagonal
    // component differ by 30%.
    expect(cell.lengths[2]! / vectors[8]!).toBeCloseTo(Math.SQRT2, 12);
  });

  it("matches OpenMM byte for byte, outside the title and the cosines", () => {
    // 🔴 The cosines are excluded from the *byte* comparison and checked
    // numerically below, because byte equality there is not achievable and
    // chasing it would mean reproducing CPython's libm. `dcdfile.py` evaluates
    // `sin(pi/2 - acos(r))`, two transcendentals that cancel to `r`, and V8's
    // `acos` and `sin` do not agree with CPython's in the last bit. Measured on
    // this fixture: four of the five frames agree exactly and one differs by a
    // single ulp, where OpenMM's round trip is the *less* accurate of the two --
    // a rhombic dodecahedron's cos(beta) is exactly 0.5 at every width, and
    // OpenMM returns 0.49999999999999983 against our 0.4999999999999999.
    //
    // Everything that is not a cosine -- the header, the edge lengths, every
    // coordinate, every record marker -- is still held to the byte, which is
    // what catches a cell written as a rectangle.
    const { ours, writer } = write();
    const theirs = new Uint8Array(readFileSync(`${triclinicDir}/reference.dcd`));
    expect(ours.length).toBe(facts!.bytes);
    const titleStart = facts!.title_offset;
    const titleEnd = titleStart + facts!.title_bytes;
    const skipped = new Set<number>();
    for (let i = titleStart; i < titleEnd; i++) skipped.add(i);
    for (let frame = 0; frame < facts!.frames; frame++) {
      // Each frame opens with the 56-byte cell record: a 4-byte marker, then
      // [a, cos gamma, b, cos beta, cos alpha, c].
      const doubles = DCD_HEADER_BYTES + frame * writer.frameBytes + 4;
      for (const slot of [1, 3, 4]) {
        for (let byte = 0; byte < 8; byte++) skipped.add(doubles + slot * 8 + byte);
      }
    }
    const differing: number[] = [];
    for (let i = 0; i < theirs.length; i++) {
      if (skipped.has(i)) continue;
      if (ours[i] !== theirs[i]) differing.push(i);
    }
    expect(
      differing.slice(0, 16),
      `${differing.length} bytes differ, first at ${differing[0]}`,
    ).toEqual([]);
  });

  it("agrees with OpenMM's cosines to within one ulp", () => {
    const { ours, writer } = write();
    const theirs = new Uint8Array(readFileSync(`${triclinicDir}/reference.dcd`));
    const mine = new DataView(ours.buffer, ours.byteOffset, ours.byteLength);
    const yours = new DataView(theirs.buffer, theirs.byteOffset, theirs.byteLength);
    let worst = 0;
    for (let frame = 0; frame < facts!.frames; frame++) {
      const doubles = DCD_HEADER_BYTES + frame * writer.frameBytes + 4;
      for (const slot of [1, 3, 4]) {
        const at = doubles + slot * 8;
        worst = Math.max(
          worst,
          Math.abs(mine.getFloat64(at, true) - yours.getFloat64(at, true)),
        );
      }
      // The lengths, by contrast, are exact -- they are arithmetic, not
      // transcendental, and the byte comparison above already covers them.
      for (const slot of [0, 2, 5]) {
        const at = doubles + slot * 8;
        expect(mine.getFloat64(at, true)).toBe(yours.getFloat64(at, true));
      }
    }
    // One ulp at 0.5 is 1.1e-16. Anything structural -- a cosine of zero where
    // OpenMM wrote 0.5 -- is 4.5e15 times this.
    expect(worst).toBeLessThan(2.3e-16);
  });

  it("round-trips the cell through the reader", () => {
    const { ours } = write();
    const read = readDcd(ours);
    const vectors = readNpy(`${triclinicDir}/boxvectors.npy`).data;
    for (let frame = 0; frame < facts!.frames; frame++) {
      const cell = unitCellOf([
        [vectors[frame * 9]!, vectors[frame * 9 + 1]!, vectors[frame * 9 + 2]!],
        [vectors[frame * 9 + 3]!, vectors[frame * 9 + 4]!, vectors[frame * 9 + 5]!],
        [vectors[frame * 9 + 6]!, vectors[frame * 9 + 7]!, vectors[frame * 9 + 8]!],
      ]);
      for (let axis = 0; axis < 3; axis++) {
        // f32 is not involved here -- the cell is f64 in the file -- so this is
        // only the angstrom conversion's round-off.
        expect(read.boxes[frame * 3 + axis]).toBeCloseTo(cell.lengths[axis]!, 9);
        expect(read.cosines[frame * 3 + axis]).toBeCloseTo(cell.cosines[axis]!, 12);
      }
    }
  });
});
