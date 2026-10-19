/**
 * A reader for the little corner of the `.npy` format the oracle tools write.
 *
 * `tools/capture_constraints.py` saves C-contiguous float64 arrays, which is a
 * six-byte magic, a version, a two-byte header length, an ASCII dict, and the
 * raw data. Supporting exactly that is twenty lines and keeps the oracle
 * self-describing, where a flat binary plus a shape in a JSON file could drift
 * apart.
 */

import { readFileSync } from "node:fs";

export interface NpyArray {
  readonly shape: readonly number[];
  readonly data: Float64Array;
}

export function readNpy(path: string): NpyArray {
  const buffer = readFileSync(path);
  const magic = buffer.subarray(0, 6).toString("latin1");
  if (magic !== "\x93NUMPY") throw new Error(`${path} is not a .npy file`);
  const major = buffer[6] as number;
  const headerLength =
    major === 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
  const headerStart = major === 1 ? 10 : 12;
  const header = buffer
    .subarray(headerStart, headerStart + headerLength)
    .toString("latin1");

  const descriptor = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  if (descriptor !== "<f8") {
    throw new Error(`${path} has dtype ${descriptor}; only little-endian f8 is read`);
  }
  if (/'fortran_order':\s*True/.test(header)) {
    throw new Error(`${path} is Fortran-ordered; only C order is read`);
  }
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeText
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map((part) => Number.parseInt(part, 10));

  const offset = headerStart + headerLength;
  const count = shape.reduce((a, b) => a * b, 1);
  // `readFileSync` may return a view into a larger pool, so the byte offset is
  // relative to the buffer's own start.
  const data = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    data[i] = buffer.readDoubleLE(offset + i * 8);
  }
  return { shape, data };
}
