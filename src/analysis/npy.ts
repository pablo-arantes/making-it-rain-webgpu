/**
 * A decoder for the little corner of the `.npy` format the oracle tools write.
 *
 * `tools/capture_constraints.py` saves C-contiguous float64 arrays, which is a
 * six-byte magic, a version, a two-byte header length, an ASCII dict, and the
 * raw data. Supporting exactly that is thirty lines and keeps the oracle
 * self-describing, where a flat binary plus a shape in a JSON file could drift
 * apart.
 *
 * It takes an `ArrayBuffer` rather than a path because both lanes need it: the
 * unit tests read the file from disk, and the validation page fetches it from
 * the dev server. test/npy.ts is the thin file-reading wrapper.
 */

export interface NpyArray {
  readonly shape: readonly number[];
  readonly data: Float64Array;
}

const MAGIC = "\x93NUMPY";

export function decodeNpy(buffer: ArrayBuffer, what = "array"): NpyArray {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  let magic = "";
  for (let i = 0; i < 6; i++) magic += String.fromCharCode(bytes[i] as number);
  if (magic !== MAGIC) throw new Error(`${what} is not a .npy file`);

  const major = bytes[6] as number;
  const headerLength =
    major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  let header = "";
  for (let i = headerStart; i < headerStart + headerLength; i++) {
    header += String.fromCharCode(bytes[i] as number);
  }

  const descriptor = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  if (descriptor !== "<f8") {
    throw new Error(`${what} has dtype ${descriptor}; only little-endian f8 is read`);
  }
  if (/'fortran_order':\s*True/.test(header)) {
    throw new Error(`${what} is Fortran-ordered; only C order is read`);
  }
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "";
  const shape = shapeText
    .split(",")
    .map((part) => part.trim())
    .filter((part) => part.length > 0)
    .map((part) => Number.parseInt(part, 10));

  const offset = headerStart + headerLength;
  const count = shape.reduce((a, b) => a * b, 1);
  // Copied element by element rather than aliased: the data offset is only
  // eight-byte aligned when the header happens to make it so, and a typed-array
  // view on a misaligned offset throws.
  const data = new Float64Array(count);
  for (let i = 0; i < count; i++) data[i] = view.getFloat64(offset + i * 8, true);
  return { shape, data };
}
