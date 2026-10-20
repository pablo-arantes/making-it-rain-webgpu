/**
 * `RunOutputs` over the Origin Private File System.
 *
 * A run writes trajectories measured in hundreds of megabytes and has to
 * survive the tab being closed, so the artefacts cannot live in memory. OPFS is
 * the only browser storage that is both large and seekable, and `RunOutputs`
 * was kept to four operations -- append, patch, write, read -- precisely so that
 * it could sit on top of it.
 *
 * 🔴 Patching is why this is not just "write a file". A DCD's frame count is not
 * known when its header goes out, so the header has to be rewritten at the end;
 * `FileSystemWritableFileStream` can do that only with `keepExistingData: true`
 * and an explicit position, because opening a writable otherwise **truncates
 * the file**. Getting that wrong does not fail -- it silently empties a
 * trajectory that took an hour to write.
 *
 * Appending needs the current size, which means a `getFile()` round trip per
 * call. That is fine at the rate this is used (a frame per reporting interval,
 * seconds apart) and would not be inside a step loop. A worker could use
 * `createSyncAccessHandle` instead and keep the handle open; this runs on the
 * main thread, where that API is unavailable.
 */

import type { RunOutputs } from "./protocol.js";

export class OpfsUnavailable extends Error {
  constructor(reason: string) {
    super(
      `the Origin Private File System is not available: ${reason}. A run's ` +
        "artefacts can reach hundreds of megabytes and have to survive the tab " +
        "closing, so there is no in-memory fallback worth offering -- a run " +
        "that cannot checkpoint is a run that cannot be resumed.",
    );
    this.name = "OpfsUnavailable";
  }
}

export interface OpfsOutputs extends RunOutputs {
  /** Every artefact written so far, with its size. */
  list(): Promise<readonly { readonly name: string; readonly bytes: number }[]>;
  /** A `Blob` of one artefact, for a download link. */
  blob(name: string): Promise<Blob>;
  /** Remove everything, so a fresh run does not resume into an old one. */
  clear(): Promise<void>;
}

/**
 * Open (or create) a directory for one job's artefacts.
 *
 * Named after the job so two runs do not resume into each other, which is the
 * failure a flat namespace would produce the first time somebody reused a job
 * name with different settings.
 */
export async function openOpfsOutputs(directory: string): Promise<OpfsOutputs> {
  if (typeof navigator === "undefined" || navigator.storage === undefined) {
    throw new OpfsUnavailable("navigator.storage is missing");
  }
  if (typeof navigator.storage.getDirectory !== "function") {
    throw new OpfsUnavailable(
      "navigator.storage.getDirectory is missing, which means this browser has " +
        "no OPFS",
    );
  }
  const root = await navigator.storage.getDirectory();
  const folder = await root.getDirectoryHandle(directory, { create: true });
  const encoder = new TextEncoder();

  const listFiles = async (): Promise<
    readonly { readonly name: string; readonly bytes: number }[]
  > => {
    const entries: { name: string; bytes: number }[] = [];
    // `FileSystemDirectoryHandle` is async-iterable, which TypeScript's DOM
    // library has not always described; the cast is to that, not to `any`.
    const iterable = folder as unknown as AsyncIterable<[string, FileSystemHandle]>;
    for await (const [name, entry] of iterable) {
      if (entry.kind !== "file") continue;
      entries.push({
        name,
        bytes: (await (entry as FileSystemFileHandle).getFile()).size,
      });
    }
    return entries.sort((a, b) => a.name.localeCompare(b.name));
  };

  const handle = async (
    name: string,
    create: boolean,
  ): Promise<FileSystemFileHandle | undefined> => {
    try {
      return await folder.getFileHandle(name, { create });
    } catch (error) {
      if (!create && (error as DOMException).name === "NotFoundError") {
        return undefined;
      }
      throw error;
    }
  };

  return {
    async append(name, bytes) {
      const file = (await handle(name, true)) as FileSystemFileHandle;
      const existing = (await file.getFile()).size;
      // `keepExistingData` and an explicit position: without the flag this
      // truncates, and without the position it would overwrite from zero.
      const stream = await file.createWritable({ keepExistingData: true });
      await stream.write({
        type: "write",
        position: existing,
        data: bytes as unknown as BufferSource,
      });
      await stream.close();
    },

    async patch(name, offset, bytes) {
      const file = await handle(name, false);
      if (file === undefined) throw new Error(`cannot patch missing ${name}`);
      const stream = await file.createWritable({ keepExistingData: true });
      await stream.write({
        type: "write",
        position: offset,
        data: bytes as unknown as BufferSource,
      });
      await stream.close();
    },

    async write(name, text) {
      const file = (await handle(name, true)) as FileSystemFileHandle;
      // Deliberately truncating: `write` replaces an artefact, and a log
      // rewritten shorter must not keep the tail of its previous self.
      const stream = await file.createWritable();
      await stream.write(encoder.encode(text) as unknown as BufferSource);
      await stream.close();
    },

    async read(name) {
      const file = await handle(name, false);
      if (file === undefined) return undefined;
      return (await file.getFile()).text();
    },

    async exists(name) {
      return (await handle(name, false)) !== undefined;
    },

    list: listFiles,

    async blob(name) {
      const file = await handle(name, false);
      if (file === undefined) throw new Error(`${name} does not exist`);
      return file.getFile();
    },

    async clear() {
      for (const { name } of await listFiles()) {
        await folder.removeEntry(name);
      }
    },
  };
}
