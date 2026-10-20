/**
 * Artefacts that outlive the device, and a run that resumes after losing one.
 *
 * A browser can take the GPU away: a driver reset, a laptop switching
 * graphics, a tab backgrounded for long enough. `device.lost` is not an error
 * condition to report, it is a normal event on a run measured in hours, and the
 * only honest answer is to re-acquire a device and carry on from the last
 * checkpoint.
 *
 * Two things are checked here, and the second is the one that matters.
 *
 * **OPFS carries the bytes.** The same protocol run against OPFS and against
 * memory must produce byte-identical artefacts. That is a sharper test than
 * "the files look plausible", and it specifically catches the trap in
 * `FileSystemWritableFileStream`: opening a writable without
 * `keepExistingData` truncates, so a header patch at the end of a run would
 * silently empty a trajectory instead of fixing two words of it.
 *
 * **A lost device costs the stage in flight and nothing else.** A device is
 * destroyed for real, a new one is acquired, and the protocol is run again over
 * the same directory -- where it skips what is already checkpointed. Stage
 * granularity is the notebook's own resume granularity, so a loss costs at most
 * one stride.
 *
 * 🔴 It acquires its **own** devices rather than taking the caller's. The first
 * version destroyed the one it was handed, and every check that ran after it on
 * the same page failed with "The Device was lost before mapping was resolved"
 * -- a validator reaching outside its own scope to break shared state. What
 * this needs is *a* device to lose, not *the* device.
 */

import { requestEngineAdapter, requestEngineDevice } from "./device.js";
import { openOpfsOutputs, type OpfsOutputs } from "../engine/opfs.js";
import { runProtocol, type ProtocolSettings } from "../engine/protocol.js";
import { memoryOutputs } from "./validate-protocol.js";

export interface PersistenceComparison {
  readonly directory: string;
  /** Artefacts in OPFS after the run, with their sizes. */
  readonly artefacts: readonly { readonly name: string; readonly bytes: number }[];
  /** Names whose OPFS bytes differ from the in-memory run's. Must be empty. */
  readonly differing: readonly string[];
  /** Names present in one sink and not the other. */
  readonly missing: readonly string[];
  /** Whether the device really was lost, rather than the test pretending. */
  readonly deviceWasLost: boolean;
  /** Stages the resumed run skipped, and the ones it had to redo. */
  readonly skippedAfterLoss: readonly string[];
  readonly redoneAfterLoss: readonly string[];
  readonly resumedSuccessfully: boolean;
  readonly ok: boolean;
}

/**
 * `runProtocol` against two sinks and a device that dies in the middle.
 *
 * `build` makes an engine factory for a given device, so the caller owns how a
 * system becomes an engine and this owns only the persistence.
 */
export async function validatePersistence(
  directory: string,
  settings: ProtocolSettings,
  build: (
    device: GPUDevice,
  ) => Promise<Parameters<typeof runProtocol>[3]>,
  system: Parameters<typeof runProtocol>[0],
): Promise<PersistenceComparison> {
  const opfs: OpfsOutputs = await openOpfsOutputs(directory);
  // A fresh run must not resume into an old one, which is what a leftover
  // directory from a previous test would make it do.
  await opfs.clear();

  const adapter = await requestEngineAdapter();
  const first = await requestEngineDevice(adapter);

  // --- The same run, to two sinks -----------------------------------------
  const memory = memoryOutputs();
  await runProtocol(system, opfs, settings, await build(first));
  await runProtocol(system, memory, settings, await build(first));

  const artefacts = await opfs.list();
  const decoder = new TextDecoder();
  const differing: string[] = [];
  const missing: string[] = [];
  for (const { name } of artefacts) {
    const theirs = memory.files.get(name);
    if (theirs === undefined) {
      missing.push(name);
      continue;
    }
    const ours = new Uint8Array(await (await opfs.blob(name)).arrayBuffer());

    // Two artefacts cannot match byte for byte, and both for the same reason:
    // they contain wall-clock time. `run.json` has timestamps and per-stage
    // speeds; a `.log` has a ns/day column, which differs between two runs of
    // the same input and whose decimal is not even the same length -- the
    // first version of this check reported three logs differing by one or two
    // bytes and nothing else.
    //
    // Skipping them would weaken the claim, so the log is compared column by
    // column with the speed left out. That says the energies, temperature and
    // volume are identical and only the stopwatch moved.
    if (name === "run.json") continue;
    if (name.endsWith(".log")) {
      const columns = (bytes: Uint8Array): string =>
        decoder
          .decode(bytes)
          .trim()
          .split("\n")
          .map((line) => line.split(",").slice(0, -1).join(","))
          .join("\n");
      if (columns(ours) !== columns(theirs)) {
        differing.push(`${name} (differs outside the speed column)`);
      }
      continue;
    }

    if (ours.length !== theirs.length) {
      differing.push(`${name} (${ours.length} vs ${theirs.length} bytes)`);
      continue;
    }
    for (let i = 0; i < ours.length; i++) {
      if (ours[i] !== theirs[i]) {
        differing.push(`${name} (first difference at byte ${i})`);
        break;
      }
    }
  }
  for (const name of memory.files.keys()) {
    if (!artefacts.some((a) => a.name === name)) missing.push(`${name} (OPFS)`);
  }

  // --- Lose the device, then resume ---------------------------------------
  const lossDirectory = `${directory}-loss`;
  const lossOutputs = await openOpfsOutputs(lossDirectory);
  await lossOutputs.clear();

  // One stride only, so the first run completes equilibration and stops there.
  const firstPass: ProtocolSettings = {
    ...settings,
    production: { ...settings.production, strides: 1 },
  };
  await runProtocol(system, lossOutputs, firstPass, await build(first));

  // Now take that GPU away for real. `device.lost` resolves on destroy, which
  // is the same signal a driver reset delivers.
  first.destroy();
  const info = await first.lost;
  const deviceWasLost = info !== undefined;

  const replacement = await requestEngineDevice(await requestEngineAdapter());
  const resumed = await runProtocol(
    system,
    lossOutputs,
    settings,
    await build(replacement),
  );
  const skippedAfterLoss = resumed.stages.filter((s) => s.skipped).map((s) => s.name);
  const redoneAfterLoss = resumed.stages.filter((s) => !s.skipped).map((s) => s.name);
  replacement.destroy();

  const resumedSuccessfully =
    // Equilibration and the completed stride are skipped; only the stride that
    // never ran is done.
    skippedAfterLoss.length === 2 &&
    redoneAfterLoss.length === settings.production.strides - 1 &&
    (await lossOutputs.exists("run.json"));

  return {
    directory,
    artefacts,
    differing,
    missing,
    deviceWasLost,
    skippedAfterLoss,
    redoneAfterLoss,
    resumedSuccessfully,
    ok:
      differing.length === 0 &&
      missing.length === 0 &&
      deviceWasLost &&
      resumedSuccessfully,
  };
}
