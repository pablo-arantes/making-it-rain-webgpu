/**
 * A DCD writer, byte-compatible with OpenMM's.
 *
 * Transcribed from `openmm/app/dcdfile.py`, which was read rather than
 * recalled -- DCD is a Fortran unformatted file with a record length written
 * before *and* after every record, several mutually incompatible dialects, and
 * a header whose fields are patched as frames arrive. Everything here that
 * looks arbitrary is CHARMM's.
 *
 * The format's three traps, all of which silently produce a file that opens and
 * reads wrong rather than one that fails:
 *
 *   * **Ångström, not nanometres.** Coordinates and box lengths are multiplied
 *     by ten. A file in nm opens fine and shows a system ten times too small.
 *   * **The timestep is in AKMA units**, so `dt / 0.04888821`. Wrong here and
 *     every time axis in every downstream analysis is wrong by a constant.
 *   * **The box record holds cosines, not angles**, in the order
 *     `[a, cos γ, b, cos β, cos α, c]` -- lengths and angles interleaved, and
 *     not in the order anybody would guess.
 *
 * `NSET` (the frame count) and `NSTEP` (the last frame's step) are not known
 * when the header is written, so OpenMM seeks back and patches them after every
 * frame. A browser has no seekable stream in the general case, so this exposes
 * the patch instead: `header()` writes placeholders, frames are appended, and
 * `headerPatches()` says which bytes to overwrite once the run has finished.
 * That works for a `Blob` assembled in memory and for an OPFS file handle
 * alike.
 */

import type { Vec3 } from "../engine/system.js";

/** One AKMA time unit, in picoseconds. The constant DCD's `dt` is scaled by. */
export const AKMA_PICOSECONDS = 0.04888821;

/** Nanometres to ångström. */
const TO_ANGSTROM = 10;

export interface DcdOptions {
  readonly particleCount: number;
  /** ps, the integrator's timestep. */
  readonly timeStep: number;
  /** The step index of the first frame. */
  readonly firstStep?: number;
  /** Steps between frames. */
  readonly interval?: number;
  /** Whether each frame carries a unit cell. False for a non-periodic system. */
  readonly periodic?: boolean;
  /**
   * Up to two 80-character lines. Longer ones are truncated, as the format
   * requires; shorter ones are padded with NULs, which is what OpenMM's
   * `struct.pack('<80s', ...)` does.
   */
  readonly titles?: readonly string[];
}

/** Where the header keeps the two fields that depend on the frame count. */
const NSET_OFFSET = 8;
const NSTEP_OFFSET = 20;

/**
 * The header's total length, which is also where the first frame begins.
 *
 * 48 bytes of `CORD` and the first nine `icntrl` words plus the timestep, 52
 * more of `icntrl` and the two record markers that follow it, 160 of title, and
 * 16 for the title record's closing marker and the atom count's own record.
 * Checked against the fixture rather than reasoned about: 2,896 bytes for five
 * frames of 37 atoms, less 5 x 524, is 276.
 */
export const DCD_HEADER_BYTES = 276;

export class DcdWriter {
  private frames = 0;
  private readonly firstStep: number;
  private readonly interval: number;
  private readonly periodic: boolean;

  constructor(private readonly options: DcdOptions) {
    this.firstStep = options.firstStep ?? 0;
    this.interval = options.interval ?? 1;
    this.periodic = options.periodic ?? true;
    if (!Number.isInteger(options.particleCount) || options.particleCount <= 0) {
      throw new Error(`a DCD needs at least one particle, got ${options.particleCount}`);
    }
  }

  get frameCount(): number {
    return this.frames;
  }

  /** How many bytes each frame occupies, which a caller may want to budget. */
  get frameBytes(): number {
    const coordinates = 3 * (8 + this.options.particleCount * 4);
    return coordinates + (this.periodic ? 56 : 0);
  }

  /**
   * The header, with `NSET` and `NSTEP` as placeholders.
   *
   * The layout, since the packing above hides it: a 4-byte record marker, then
   * 84 bytes of `CORD` plus twenty `icntrl` words, then its closing marker;
   * then a 164-byte title record; then a 4-byte record holding the atom count.
   */
  header(): Uint8Array {
    const bytes = new ArrayBuffer(DCD_HEADER_BYTES);
    const view = new DataView(bytes);
    const u8 = new Uint8Array(bytes);
    let at = 0;
    const int = (value: number): void => {
      view.setInt32(at, value, true);
      at += 4;
    };
    const float = (value: number): void => {
      view.setFloat32(at, value, true);
      at += 4;
    };

    int(84); // opening record marker
    u8[4] = 0x43; // C
    u8[5] = 0x4f; // O
    u8[6] = 0x52; // R
    u8[7] = 0x44; // D
    at = 8;
    int(0); // icntrl[0] NSET, patched
    int(this.firstStep); // icntrl[1] ISTART
    int(this.interval); // icntrl[2] NSAVC
    int(0); // icntrl[3] NSTEP, patched
    for (let i = 0; i < 5; i++) int(0); // icntrl[4..8]
    float(this.options.timeStep / AKMA_PICOSECONDS); // icntrl[9] DELTA
    int(this.periodic ? 1 : 0); // icntrl[10] unit cell present
    for (let i = 0; i < 8; i++) int(0); // icntrl[11..18]
    int(24); // icntrl[19]: the CHARMM version, which selects this dialect
    int(84); // closing record marker

    int(164); // title record, opening marker
    const titles = this.options.titles ?? ["Created by making-it-rain-webgpu"];
    int(2); // NTITLE: always two, as OpenMM writes
    for (let line = 0; line < 2; line++) {
      const text = titles[line] ?? "";
      for (let i = 0; i < 80; i++) {
        // Truncated at 80 and NUL-padded, which is `struct.pack('<80s', ...)`.
        const code = i < text.length ? text.charCodeAt(i) : 0;
        u8[at + i] = code < 128 ? code : 0x3f; // '?' for anything not ASCII
      }
      at += 80;
    }
    int(164); // closing marker

    int(4); // the atom count's own record
    int(this.options.particleCount);
    int(4);
    return u8;
  }

  /**
   * One frame.
   *
   * `positions` is nm, xyz interleaved; `box` is the three edge lengths in nm.
   * Both are converted to ångström here, which is the single most common way to
   * get a DCD wrong.
   */
  frame(positions: Float64Array, box?: Vec3): Uint8Array {
    const n = this.options.particleCount;
    if (positions.length !== n * 3) {
      throw new Error(
        `expected ${n * 3} coordinates for ${n} particles, got ${positions.length}`,
      );
    }
    if (this.periodic && box === undefined) {
      throw new Error(
        "a periodic DCD needs a box for every frame; the flag in the header " +
          "says each frame carries one, and a reader will try to read it",
      );
    }
    for (let i = 0; i < positions.length; i++) {
      if (!Number.isFinite(positions[i] as number)) {
        throw new Error(
          `coordinate ${i} is ${positions[i]}. OpenMM refuses to write a ` +
            "non-finite position and so does this: a trajectory with a NaN in " +
            "it reads as a valid file and ruins every analysis downstream.",
        );
      }
    }

    const bytes = new ArrayBuffer(this.frameBytes);
    const view = new DataView(bytes);
    let at = 0;
    if (this.periodic) {
      const lengths = box as Vec3;
      view.setInt32(at, 48, true);
      at += 4;
      // [a, cos gamma, b, cos beta, cos alpha, c]. All three angles are right
      // angles for a rectangular box, so the cosines are zero -- the engine
      // refuses triclinic boxes upstream (`TriclinicBoxUnsupported`), and this
      // is where that assumption is written down.
      const cell = [
        (lengths[0] as number) * TO_ANGSTROM,
        0,
        (lengths[1] as number) * TO_ANGSTROM,
        0,
        0,
        (lengths[2] as number) * TO_ANGSTROM,
      ];
      for (const value of cell) {
        view.setFloat64(at, value, true);
        at += 8;
      }
      view.setInt32(at, 48, true);
      at += 4;
    }

    const marker = 4 * n;
    for (let axis = 0; axis < 3; axis++) {
      view.setInt32(at, marker, true);
      at += 4;
      for (let i = 0; i < n; i++) {
        view.setFloat32(at, (positions[i * 3 + axis] as number) * TO_ANGSTROM, true);
        at += 4;
      }
      view.setInt32(at, marker, true);
      at += 4;
    }

    this.frames++;
    return new Uint8Array(bytes);
  }

  /**
   * The header bytes that have to be rewritten now the frame count is known.
   *
   * Two four-byte words. A caller holding the whole file in memory can splice
   * them in; one writing to a seekable handle can seek to each offset.
   */
  headerPatches(): readonly { readonly offset: number; readonly bytes: Uint8Array }[] {
    const word = (value: number): Uint8Array => {
      const bytes = new ArrayBuffer(4);
      new DataView(bytes).setInt32(0, value, true);
      return new Uint8Array(bytes);
    };
    return [
      { offset: NSET_OFFSET, bytes: word(this.frames) },
      {
        offset: NSTEP_OFFSET,
        // The step the last frame was written at, which is what OpenMM patches
        // here after every frame.
        bytes: word(this.firstStep + Math.max(0, this.frames - 1) * this.interval),
      },
    ];
  }

  /** Apply `headerPatches` to a buffer that begins with the header. */
  patch(file: Uint8Array): void {
    for (const { offset, bytes } of this.headerPatches()) file.set(bytes, offset);
  }
}

export interface DcdTrajectory {
  readonly particleCount: number;
  readonly frameCount: number;
  /** The step the first frame was written at. */
  readonly firstStep: number;
  /** Steps between frames. */
  readonly interval: number;
  /** ps */
  readonly timeStep: number;
  readonly periodic: boolean;
  /** nm, `frameCount * particleCount * 3`, xyz interleaved per frame. */
  readonly positions: Float32Array;
  /** nm, three edge lengths per frame; empty when the file has no boxes. */
  readonly boxes: Float64Array;
}

export class DcdFormatError extends Error {
  constructor(detail: string) {
    super(
      `this does not look like a DCD this reader can handle: ${detail}. ` +
        "The format is a Fortran-record file and the records carry their own " +
        "lengths, so a mismatch is caught here rather than producing " +
        "plausible coordinates.",
    );
    this.name = "DcdFormatError";
  }
}

/**
 * Read a DCD back.
 *
 * The writer existed first, and for a while that was enough: trajectories went
 * out to a file and other people's tools read them. The analyses changed that
 * — Cα RMSD and the rest run over a trajectory this code wrote, so it has to
 * be able to read one.
 *
 * Deliberately strict. Every Fortran record carries its own byte count twice,
 * once before and once after, and this checks both. A DCD read with the wrong
 * endianness, or with the unit-cell flag misjudged, does not fail — it yields
 * coordinates that are merely wrong, which is the failure mode that ruins an
 * analysis quietly. The markers are the only thing that distinguishes the two
 * cases, so they are checked rather than skipped.
 *
 * Only the little-endian, CHARMM-24 dialect OpenMM writes is accepted. A
 * big-endian or pre-CHARMM file is refused by name instead of being
 * misinterpreted.
 */
export function readDcd(file: Uint8Array): DcdTrajectory {
  const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
  if (file.byteLength < DCD_HEADER_BYTES) {
    throw new DcdFormatError(`only ${file.byteLength} bytes, less than a header`);
  }

  const first = view.getInt32(0, true);
  if (first !== 84) {
    // A big-endian file has 84 in the other byte order, which is a huge
    // little-endian number -- worth naming, since it is the common case.
    const swapped = view.getInt32(0, false);
    throw new DcdFormatError(
      swapped === 84
        ? "the leading record marker is big-endian; only the little-endian " +
          "dialect OpenMM writes is supported"
        : `the leading record marker is ${first}, not 84`,
    );
  }
  const magic = String.fromCharCode(file[4]!, file[5]!, file[6]!, file[7]!);
  if (magic !== "CORD") throw new DcdFormatError(`the magic is "${magic}", not "CORD"`);

  const frameCount = view.getInt32(8, true);
  const firstStep = view.getInt32(12, true);
  const interval = view.getInt32(16, true);
  const delta = view.getFloat32(44, true);
  const periodic = view.getInt32(48, true) !== 0;
  const charmm = view.getInt32(84, true);
  if (charmm === 0) {
    throw new DcdFormatError(
      "the CHARMM version word is zero, which marks an X-PLOR style file whose " +
        "header is laid out differently",
    );
  }
  if (view.getInt32(88, true) !== 84) {
    throw new DcdFormatError("the header's closing record marker is not 84");
  }

  // The title record, whose length is declared rather than assumed: OpenMM
  // writes two 80-character lines, but the format allows any number.
  const titleBytes = view.getInt32(92, true);
  let at = 96 + titleBytes;
  if (view.getInt32(at, true) !== titleBytes) {
    throw new DcdFormatError("the title record's two markers disagree");
  }
  at += 4;

  if (view.getInt32(at, true) !== 4) {
    throw new DcdFormatError("the atom-count record is not four bytes long");
  }
  const particleCount = view.getInt32(at + 4, true);
  if (particleCount <= 0) throw new DcdFormatError(`${particleCount} particles`);
  at += 12;

  const coordinateBytes = 4 * particleCount;
  const perFrame = 3 * (coordinateBytes + 8) + (periodic ? 56 : 0);
  const available = file.byteLength - at;
  const frames = Math.floor(available / perFrame);
  if (frames === 0) {
    throw new DcdFormatError(
      `no complete frame: ${available} bytes left but a frame needs ${perFrame}`,
    );
  }

  // 🔴 The header's frame count is trusted only as far as the file supports.
  // A run killed mid-write leaves NSET claiming more frames than were flushed,
  // and reading past the end would fabricate the last one. OpenMM's own
  // reporter patches NSET after each frame, so the two normally agree.
  const count = frameCount > 0 ? Math.min(frameCount, frames) : frames;

  const positions = new Float32Array(count * particleCount * 3);
  const boxes = periodic ? new Float64Array(count * 3) : new Float64Array(0);
  const fromAngstrom = 1 / TO_ANGSTROM;

  for (let frame = 0; frame < count; frame++) {
    if (periodic) {
      if (view.getInt32(at, true) !== 48) {
        throw new DcdFormatError(
          `frame ${frame}'s unit-cell record is not 48 bytes; the header says ` +
            "each frame carries a box",
        );
      }
      at += 4;
      // [a, cos gamma, b, cos beta, cos alpha, c].
      const a = view.getFloat64(at, true);
      const b = view.getFloat64(at + 16, true);
      const c = view.getFloat64(at + 40, true);
      boxes[frame * 3] = a * fromAngstrom;
      boxes[frame * 3 + 1] = b * fromAngstrom;
      boxes[frame * 3 + 2] = c * fromAngstrom;
      at += 48;
      if (view.getInt32(at, true) !== 48) {
        throw new DcdFormatError(`frame ${frame}'s unit-cell record markers disagree`);
      }
      at += 4;
    }
    for (let axis = 0; axis < 3; axis++) {
      if (view.getInt32(at, true) !== coordinateBytes) {
        throw new DcdFormatError(
          `frame ${frame} axis ${axis}: the record marker is ` +
            `${view.getInt32(at, true)}, not ${coordinateBytes}. The atom count ` +
            "in the header and the frame records do not agree",
        );
      }
      at += 4;
      const base = frame * particleCount * 3;
      for (let i = 0; i < particleCount; i++) {
        positions[base + i * 3 + axis] = view.getFloat32(at, true) * fromAngstrom;
        at += 4;
      }
      if (view.getInt32(at, true) !== coordinateBytes) {
        throw new DcdFormatError(`frame ${frame} axis ${axis}: markers disagree`);
      }
      at += 4;
    }
  }

  return {
    particleCount,
    frameCount: count,
    firstStep,
    interval,
    timeStep: delta * AKMA_PICOSECONDS,
    periodic,
    positions,
    boxes,
  };
}
