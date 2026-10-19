/**
 * Reading a `.npy` fixture from disk.
 *
 * The decoder itself is src/analysis/npy.ts, shared with the validation page,
 * which fetches the same files over HTTP instead.
 */

import { readFileSync } from "node:fs";
import { decodeNpy, type NpyArray } from "../src/analysis/npy.js";

export type { NpyArray };

export function readNpy(path: string): NpyArray {
  const buffer = readFileSync(path);
  // `readFileSync` may return a view into a larger pool, so the view's own
  // range has to be sliced out before it is treated as a whole buffer.
  return decodeNpy(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength) as ArrayBuffer,
    path,
  );
}
