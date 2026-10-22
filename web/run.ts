/**
 * The run page: upload a system, check it, run it, download what comes out.
 *
 * Phase 2's deliverable. Everything it does is already tested elsewhere -- the
 * protocol in Node against a stub, the kernels against OpenMM, the artefacts
 * against three trajectory readers -- so this file is orchestration and
 * presentation, and the things worth commenting are the ones that are about the
 * browser rather than about molecular dynamics.
 *
 * Three of those, in the order they bite:
 *
 *   * **A run must survive the screen sleeping.** A wake lock is requested and
 *     re-requested on visibility change, because the browser drops it when the
 *     tab is hidden and does not give it back by itself.
 *   * **A run must survive losing the GPU.** `device.lost` is normal on a
 *     multi-hour run; the answer is to re-acquire and continue from the last
 *     checkpoint, which the protocol already does by skipping finished stages.
 *   * **The UI must not block the queue.** Reports arrive from the loop's own
 *     callback, and plots are redrawn from a `requestAnimationFrame` rather
 *     than per report, so a dense reporting interval cannot starve the run.
 */

import {
  onDeviceLost,
  requestEngineAdapter,
  requestEngineDevice,
  WebGpuUnavailable,
} from "../src/gpu/device.js";
import { parseSystem } from "../src/engine/system.js";
import { deviceFitness } from "../src/gpu/device.js";
import { openOpfsOutputs, type OpfsOutputs } from "../src/engine/opfs.js";
import { runProtocol, stageNames, type ProtocolSettings } from "../src/engine/protocol.js";
import { protocolHooksFor } from "../src/gpu/validate-protocol.js";
import { verifySystem, type VerifyReport } from "../src/gpu/verify.js";
import { drawPlot, type PlotSeries } from "./plot.js";

declare global {
  interface Window {
    /** Playwright drives the page through these. */
    __mirRun?: {
      readonly state: "idle" | "verifying" | "running" | "done" | "failed";
      readonly stages?: readonly string[];
      readonly artefacts?: readonly { readonly name: string; readonly bytes: number }[];
      readonly error?: string;
      readonly reports?: number;
    };
    __mirVerify?: VerifyReport | { error: string };
  }
}

const el = <T extends HTMLElement>(id: string): T =>
  document.querySelector<T>(`#${id}`) as T;

const uploads = {
  system: el<HTMLInputElement>("file-system"),
  state: el<HTMLInputElement>("file-state"),
  restrained: el<HTMLInputElement>("file-restrained"),
  pdb: el<HTMLInputElement>("file-pdb"),
};
const status = el<HTMLElement>("run-status");
const summary = el<HTMLElement>("upload-summary");
const verifyBody = el<HTMLElement>("verify-body");
const progress = el<HTMLProgressElement>("run-progress");
const progressText = el<HTMLElement>("run-progress-text");
const artefactBody = el<HTMLElement>("artefact-body");
const startButton = el<HTMLButtonElement>("start");
const verifyButton = el<HTMLButtonElement>("verify");

interface Loaded {
  systemXml: string;
  stateXml: string;
  restrainedXml?: string;
  topologyPdb?: string;
}
const loaded: Loaded = { systemXml: "", stateXml: "" };

const series: Record<string, PlotSeries & { points: { x: number; y: number }[] }> = {
  energy: {
    label: "potential energy",
    unit: "kJ/mol",
    colour: "#2b6cb0",
    points: [],
  },
  temperature: { label: "temperature", unit: "K", colour: "#c05621", points: [] },
};
let redrawQueued = false;
let reports = 0;

function queueRedraw(): void {
  if (redrawQueued) return;
  redrawQueued = true;
  // Coalesced: a dense reporting interval must not turn into a redraw per
  // report and starve the run of its own main thread.
  requestAnimationFrame(() => {
    redrawQueued = false;
    drawPlot(el<HTMLCanvasElement>("plot-energy"), series.energy!);
    drawPlot(el<HTMLCanvasElement>("plot-temperature"), series.temperature!);
  });
}

function setStatus(text: string, kind: "" | "ok" | "bad" = ""): void {
  status.textContent = text;
  status.className = `status ${kind}`;
}

/** Human-readable remaining time, which is the number people actually watch. */
function describeRemaining(done: number, total: number, elapsed: number): string {
  if (done === 0) return "estimating…";
  const remaining = (elapsed / done) * (total - done);
  if (remaining < 90) return `${Math.round(remaining)} s left`;
  if (remaining < 5400) return `${Math.round(remaining / 60)} min left`;
  return `${(remaining / 3600).toFixed(1)} h left`;
}

async function readFiles(): Promise<boolean> {
  const read = async (input: HTMLInputElement): Promise<string | undefined> => {
    const file = input.files?.[0];
    return file === undefined ? undefined : file.text();
  };
  const systemXml = await read(uploads.system);
  const stateXml = await read(uploads.state);
  if (systemXml === undefined || stateXml === undefined) {
    summary.innerHTML =
      '<p class="note">A <code>system.xml</code> and a <code>state.xml</code> ' +
      "are the minimum: the engine's contract is OpenMM's own serialisation, so " +
      "anything OpenMM can write, this can read.</p>";
    return false;
  }
  loaded.systemXml = systemXml;
  loaded.stateXml = stateXml;
  const restrained = await read(uploads.restrained);
  if (restrained !== undefined) loaded.restrainedXml = restrained;
  const pdb = await read(uploads.pdb);
  if (pdb !== undefined) loaded.topologyPdb = pdb;

  try {
    const system = parseSystem(systemXml);
    summary.innerHTML =
      `<p class="note"><strong>${system.particleCount.toLocaleString("en-US")}</strong> ` +
      `particles, ${system.constraintCount.toLocaleString("en-US")} constraints, ` +
      `forces: ${system.forces.map((f) => f.type).join(", ")}.` +
      (loaded.restrainedXml === undefined
        ? " No restrained system given, so equilibration runs unrestrained — " +
          "the notebook restrains heavy atoms, and building that mask needs a " +
          "topology the engine does not model yet."
        : " Equilibration will use the restrained system.") +
      (loaded.topologyPdb === undefined
        ? " No PDB, so no final-frame files."
        : " Final frames will be written as PDB.") +
      "</p>";
    // 🔴 The status line is set once, at start-up, and said "Nothing is
    // uploaded." for the rest of the session however many files you loaded.
    // That is a one-line UI bug with a disproportionate cost: it appears in
    // Playwright's failure snapshots, where it reads as proof that the upload
    // never happened, and it sent one investigation of a timed-out run down
    // the wrong path entirely. A status line that can be stale is worse than
    // no status line.
    // 🔴 Checked at upload, before any GPU work. The limits that bind are per
    // device, so a system prepared on one machine can exceed another's — and
    // without this the first sign is an allocation failure part-way through a
    // run, which reads as a bug rather than as a system too large for the
    // hardware. `loadedLimits` is filled in once the adapter is known.
    if (loadedLimits !== undefined) {
      const fitness = deviceFitness(loadedLimits, system.particleCount);
      if (!fitness.fits) {
        summary.innerHTML +=
          '<p class="bad-box"><strong>This system is too large for this GPU.</strong> ' +
          fitness.problems.map((problem) => escapeText(problem.detail)).join(" ") +
          "</p>";
        setStatus("The system does not fit this GPU; see the panel.", "bad");
        return false;
      }
      summary.innerHTML +=
        `<p class="note">This GPU holds about ` +
        `${fitness.maxAtoms.toLocaleString("en-US")} atoms, limited by its ` +
        `${escapeText(fitness.limitedBy)}.</p>`;
      // The neighbour list is the difference between about 26 and about 14
      // ns/day on a solvated protein, so a run that cannot have it should say
      // so rather than just being slow. It is not a correctness note: the
      // fallback is the kernel the differential tests run, held to bit-for-bit
      // agreement with the list by `validateNeighbours`.
      if (!fitness.neighbourList.fits) {
        summary.innerHTML +=
          `<p class="note">Too large for a neighbour list on this GPU, which ` +
          `needs ${(fitness.neighbourList.bytes / 1048576).toFixed(0)} MB and ` +
          `fits about ` +
          `${fitness.neighbourList.maxAtoms.toLocaleString("en-US")} atoms. The ` +
          `run will use the cell-shell scan instead: the same forces, roughly ` +
          `half the speed.</p>`;
      }
    }

    setStatus(
      `${system.particleCount.toLocaleString("en-US")} particles loaded and ready to run.`,
      "ok",
    );
    return true;
  } catch (error) {
    summary.innerHTML = `<p class="bad-box">${
      error instanceof Error ? error.message : String(error)
    }</p>`;
    return false;
  }
}

function settingsFromForm(): ProtocolSettings {
  const number = (id: string): number => Number(el<HTMLInputElement>(id).value);
  const timeStep = number("dt") / 1000; // the field is fs, the engine is ps
  const common = {
    timeStep,
    temperature: number("temperature"),
    pressure: number("pressure"),
    trajectoryInterval: number("traj-interval"),
    logInterval: number("log-interval"),
  };
  return {
    jobName: el<HTMLInputElement>("job").value || "run",
    seed: number("seed"),
    equilibration: {
      ...common,
      nanoseconds: number("equil-ns"),
      minimisationSteps: number("min-steps"),
      restraintForceConstant: number("restraint-k"),
    },
    production: {
      ...common,
      nanoseconds: number("stride-ns"),
      strides: number("strides"),
    },
  };
}

/** Keep the screen awake, and get the lock back when the tab returns. */
async function holdWakeLock(): Promise<() => void> {
  type Sentinel = { release(): Promise<void>; released: boolean };
  const api = (
    navigator as unknown as {
      wakeLock?: { request(type: "screen"): Promise<Sentinel> };
    }
  ).wakeLock;
  if (api === undefined) return () => {};
  let sentinel: Sentinel | undefined;
  const acquire = async (): Promise<void> => {
    try {
      sentinel = await api.request("screen");
    } catch {
      // Denied, or the document is hidden. Not worth interrupting a run over.
    }
  };
  const onVisible = (): void => {
    // 🔴 The browser drops the lock when the tab is hidden and does not give it
    // back. Without this, a run minimised for ten minutes loses the screen and
    // then the GPU on a laptop that decides to sleep.
    if (document.visibilityState === "visible" && sentinel?.released !== false) {
      void acquire();
    }
  };
  await acquire();
  document.addEventListener("visibilitychange", onVisible);
  return () => {
    document.removeEventListener("visibilitychange", onVisible);
    void sentinel?.release().catch(() => {});
  };
}

async function showArtefacts(outputs: OpfsOutputs): Promise<void> {
  const files = await outputs.list();
  window.__mirRun = { ...(window.__mirRun ?? { state: "running" }), artefacts: files };
  if (files.length === 0) {
    artefactBody.innerHTML = '<p class="note">Nothing written yet.</p>';
    return;
  }
  artefactBody.innerHTML =
    "<table><tr><th>File</th><th>Size</th><th></th></tr>" +
    files
      .map(
        ({ name, bytes }) => `<tr><td><code>${name}</code></td>
          <td class="num">${(bytes / 1024).toFixed(1)} kB</td>
          <td><button data-download="${name}">download</button></td></tr>`,
      )
      .join("") +
    "</table>";
  for (const button of artefactBody.querySelectorAll<HTMLButtonElement>(
    "button[data-download]",
  )) {
    button.addEventListener("click", async () => {
      const name = button.dataset["download"] as string;
      const url = URL.createObjectURL(await outputs.blob(name));
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.click();
      // Revoked on the next tick: revoking immediately races the download in
      // some browsers.
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    });
  }
}

async function verify(): Promise<void> {
  if (!(await readFiles())) return;
  verifyButton.disabled = true;
  verifyBody.innerHTML =
    '<p class="note">Evaluating the whole force field twice — once on your GPU ' +
    "and once in double precision on the CPU. The second is slow by design; " +
    "it is the thing being compared against.</p>";
  window.__mirRun = { state: "verifying" };
  try {
    const device = await requestEngineDevice(await requestEngineAdapter());
    try {
      const report = await verifySystem(device, loaded);
      window.__mirVerify = report;
      verifyBody.innerHTML = `<table>
        <tr><th>Particles</th><td class="num">${report.particleCount.toLocaleString("en-US")}</td></tr>
        <tr><th>Forces</th><td>${report.forces.join(", ")}</td></tr>
        <tr><th>Box</th><td class="num">${report.box.map((v) => v.toFixed(3)).join(" × ")} nm</td></tr>
        <tr><th>Cutoff, α, mesh</th><td class="num">${report.cutoff} nm, ${report.alpha.toFixed(4)} nm⁻¹, ${report.mesh.join("×")}</td></tr>
        <tr><th>Energy, device vs f64</th><td class="num">${report.deviceEnergy.toFixed(3)} vs ${report.referenceEnergy.toFixed(3)} kJ/mol (${report.energyRelativeError.toExponential(2)})</td></tr>
        <tr><th>Force relative RMS</th><td class="num">${report.forceRelativeRms.toExponential(2)} of ${report.maxReferenceForce.toFixed(0)} max</td></tr>
        <tr><th>Molecules intact</th><td>${report.moleculesIntact ? "yes" : "<strong>no — see below</strong>"}, longest bonded ${report.worstBondedSeparation.toFixed(3)} nm</td></tr>
        <tr><th>Can run NPT</th><td>${report.canRunNpt ? "yes" : "<strong>no</strong>"}, cell headroom ${(report.cellHeadroom * 100).toFixed(1)}%</td></tr>
        <tr><th></th><td><span class="badge ${report.ok ? "ok" : "bad"}">${report.ok ? "pass" : "fail"}</span> in ${report.elapsed.toFixed(1)} s</td></tr>
      </table>
      ${
        report.canRunNpt
          ? ""
          : '<p class="note">A barostat needs the box to stay at least three ' +
            "cutoffs wide; this one would exhaust that. Equilibration and " +
            "production will run at constant volume. Solvate with more padding " +
            "or shorten the cutoff to get NPT.</p>"
      }
      ${
        report.moleculesIntact
          ? ""
          : '<p class="bad-box">Some molecules are split across the periodic ' +
            "boundary. Bonded forces use plain differences, as OpenMM's do, so " +
            "this state would read a bond as a box length. Re-export without " +
            "atom-wise wrapping.</p>"
      }`;
    } finally {
      device.destroy();
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    window.__mirVerify = { error: message };
    verifyBody.innerHTML = `<p class="bad-box">${message}</p>`;
  } finally {
    verifyButton.disabled = false;
  }
}

async function start(): Promise<void> {
  if (!(await readFiles())) return;
  startButton.disabled = true;
  series.energy!.points.length = 0;
  series.temperature!.points.length = 0;
  reports = 0;
  queueRedraw();

  const settings = settingsFromForm();
  const names = stageNames(settings);
  const releaseWakeLock = await holdWakeLock();
  const began = performance.now();
  let totalSteps = 0;
  let doneSteps = 0;

  try {
    const outputs = await openOpfsOutputs(`mir-${settings.jobName}`);
    await showArtefacts(outputs);
    window.__mirRun = { state: "running", stages: [names.equilibration, ...names.strides] };

    // A lost device is normal on a long run: re-acquire and let the protocol
    // skip what is already checkpointed. Up to three attempts, because a
    // device that cannot be held twice is not going to be held a fourth time.
    for (let attempt = 1; attempt <= 3; attempt++) {
      const device = await requestEngineDevice(await requestEngineAdapter());
      let lost = false;
      onDeviceLost(device, () => {
        lost = true;
      });
      try {
        const { hooks, ensemble } = await protocolHooksFor(
          device,
          {
            systemXml: loaded.systemXml,
            restrainedXml: loaded.restrainedXml ?? loaded.systemXml,
            stateXml: loaded.stateXml,
            ...(loaded.topologyPdb === undefined
              ? {}
              : { topologyPdb: loaded.topologyPdb }),
          },
          settings,
        );
        setStatus(
          `Running ${ensemble}${attempt > 1 ? ` (attempt ${attempt} after a lost device)` : ""}…`,
        );
        const record = await runProtocol(parseSystem(loaded.systemXml), outputs, settings, {
          ...hooks,
          onStage: async () => {
            await showArtefacts(outputs);
          },
          onReport: (row) => {
            reports++;
            totalSteps = Math.max(totalSteps, row.totalSteps);
            doneSteps = row.step;
            series.energy!.points.push({ x: row.step, y: row.potentialEnergy });
            series.temperature!.points.push({ x: row.step, y: row.temperature });
            progress.max = row.totalSteps;
            progress.value = row.step;
            progressText.textContent =
              `step ${row.step.toLocaleString("en-US")} of ` +
              `${row.totalSteps.toLocaleString("en-US")} · ` +
              `${row.nsPerDay.toFixed(1)} ns/day · ` +
              describeRemaining(doneSteps, totalSteps, (performance.now() - began) / 1000);
            window.__mirRun = { ...(window.__mirRun ?? { state: "running" }), reports };
            queueRedraw();
          },
        });
        await showArtefacts(outputs);
        setStatus(
          `Done: ${record.stages.filter((s) => !s.skipped).length} stage(s) run, ` +
            `${record.stages.filter((s) => s.skipped).length} already complete.`,
          "ok",
        );
        window.__mirRun = {
          ...(window.__mirRun ?? { state: "running" }),
          state: "done",
          artefacts: await outputs.list(),
        };
        return;
      } catch (error) {
        device.destroy();
        if (!lost || attempt === 3) throw error;
        setStatus("The GPU was lost. Re-acquiring and resuming from the last checkpoint…");
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(message, "bad");
    window.__mirRun = { ...(window.__mirRun ?? { state: "running" }), state: "failed", error: message };
  } finally {
    releaseWakeLock();
    startButton.disabled = false;
  }
}

for (const input of Object.values(uploads)) {
  input.addEventListener("change", () => void readFiles());
}
verifyButton.addEventListener("click", () => void verify());
startButton.addEventListener("click", () => void start());
window.addEventListener("resize", queueRedraw);
queueRedraw();
window.__mirRun = { state: "idle" };

/** The adapter's limits, for the upload-time fitness check. */
let loadedLimits: Readonly<Record<string, number>> | undefined;

const escapeText = (value: string): string =>
  value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

void (async () => {
  try {
    const adapter = await requestEngineAdapter();
    const device = await requestEngineDevice(adapter);
    const info = adapter.info;
    // 🔴 The **device's** limits, not the adapter's, and the difference is
    // large in both directions. `adapter.limits` reports what the hardware
    // could support — 4.29 GB of storage binding on this machine — while a
    // device created without asking for anything gets WebGPU's defaults, which
    // are 128 MB. Measured here: a preflight reading the adapter's maxima said
    // 4,473,924 atoms and one reading an unrequested device's said 139,810,
    // for the same GPU. Only the limits the engine actually requests describe
    // what a run can do, which is why this is taken from the device
    // `requestEngineDevice` returns.
    loadedLimits = Object.fromEntries(
      (
        [
          "maxBufferSize",
          "maxStorageBufferBindingSize",
          "maxComputeWorkgroupStorageSize",
          "maxComputeInvocationsPerWorkgroup",
          "maxComputeWorkgroupSizeX",
          "maxComputeWorkgroupsPerDimension",
          "maxStorageBuffersPerShaderStage",
        ] as const
      ).map((name) => [name, device.limits[name] as number]),
    );
    setStatus(
      `Ready on ${[info?.vendor, info?.architecture, info?.device]
        .filter(Boolean)
        .join(" · ") || "an unnamed adapter"}. Nothing is uploaded.`,
    );
    device.destroy();
  } catch (error) {
    setStatus(
      error instanceof WebGpuUnavailable
        ? error.message
        : `No usable GPU: ${error instanceof Error ? error.message : String(error)}`,
      "bad",
    );
    startButton.disabled = true;
    verifyButton.disabled = true;
  }
})();
