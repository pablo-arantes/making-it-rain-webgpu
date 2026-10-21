/**
 * The analysis page: a trajectory in, the notebook's figures out.
 *
 * `src/analysis/protein.ts` holds the analyses and is verified against NumPy
 * and MDTraj headlessly. This file is presentation, and the parts worth
 * commenting are the ones that are decisions rather than plumbing.
 *
 * **A DCD carries no names.** It is coordinates and a box, nothing else, so
 * the Cα atoms cannot be found without the topology. That is why the page asks
 * for two files and refuses to analyse with one — guessing which atoms are Cα
 * from coordinates alone is not possible, and silently analysing *all* atoms
 * would answer a different question.
 *
 * **Everything runs on the main thread.** Pairwise RMSD is O(frames²)
 * superpositions and PCA is a Jacobi solve on a `3N x 3N` covariance, so a
 * long trajectory of a large protein is seconds of blocking work. The status
 * line is updated and the frame yielded between stages so the page says what
 * it is doing; a worker is the real answer and is not written.
 */

import { readDcd, type DcdTrajectory } from "../src/analysis/dcd.js";
import { readPdb, type PdbTopology } from "../src/analysis/pdb.js";
import {
  caIndices,
  crossCorrelation,
  kernelDensity,
  matrixCsv,
  pairwiseRmsd,
  pca,
  radiusOfGyration,
  rmsdSeries,
  rmsf,
  seriesCsv,
} from "../src/analysis/protein.js";
import {
  centreOfMassSeries,
  extractFrame,
  ligandHeavyAtoms,
  ligandRmsd,
  pocketResidues,
  residueDistances,
} from "../src/analysis/ligand.js";
import { writePdb } from "../src/analysis/pdb.js";
import { drawHeatmap } from "./heatmap.js";
import { drawPlot } from "./plot.js";

declare global {
  interface Window {
    __mirAnalyse?: {
      state: "idle" | "analysing" | "done" | "failed";
      error?: string;
      summary?: Record<string, number | string>;
      artefacts?: readonly { name: string; bytes: number }[];
    };
  }
}

function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`no element #${id}`);
  return found as T;
}

const status = el<HTMLParagraphElement>("analyse-status");
const summary = el<HTMLDivElement>("trajectory-summary");
const downloads = el<HTMLDivElement>("downloads");
const button = el<HTMLButtonElement>("analyse");

function setStatus(text: string, kind: "" | "ok" | "bad" = ""): void {
  status.textContent = text;
  status.className = kind === "" ? "status" : `status ${kind}`;
}
const yieldToPaint = (): Promise<void> =>
  new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
const escape = (value: string): string =>
  value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

let trajectory: DcdTrajectory | undefined;
let topology: PdbTopology | undefined;

/** Residue names that are neither polymer nor solvent: ligand candidates. */
const SOLVENT_OR_ION = new Set([
  "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "ZN", "MN", "FE", "BR", "IOD",
  "LI", "RB", "CS", "CA", "SO4", "PO4",
]);
const POLYMER_OR_SOLVENT = new Set([
  "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU",
  "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HID", "HIE",
  "HIP", "CYX", "ASH", "GLH", "LYN", "ACE", "NME", "NHE", "MSE",
  ...SOLVENT_OR_ION,
]);

function fillLigandPicker(): void {
  const section = el<HTMLElement>("ligand-section");
  const select = el<HTMLSelectElement>("ligand-residue");
  select.innerHTML = "";
  if (topology === undefined) {
    section.hidden = true;
    return;
  }
  const counts = new Map<string, number>();
  for (const residue of topology.residues) {
    const name = residue.name.toUpperCase();
    if (POLYMER_OR_SOLVENT.has(name)) continue;
    counts.set(name, (counts.get(name) ?? 0) + residue.atoms.length);
  }
  // Largest first: the interesting ligand is almost always the biggest
  // non-polymer thing in the file.
  const ranked = [...counts].sort((a, b) => b[1] - a[1]);
  section.hidden = ranked.length === 0;
  for (const [name, atoms] of ranked) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name} — ${atoms} atoms`;
    select.append(option);
  }
}

function describe(): void {
  const parts: string[] = [];
  if (trajectory !== undefined) {
    parts.push(
      `<strong>${trajectory.frameCount}</strong> frames of ` +
        `<strong>${trajectory.particleCount.toLocaleString("en-US")}</strong> atoms, ` +
        `every ${trajectory.interval} steps ` +
        `(${(trajectory.interval * trajectory.timeStep).toFixed(2)} ps)`,
    );
  }
  if (topology !== undefined) {
    const ca = caIndices(topology);
    parts.push(
      `topology with ${topology.atoms.length.toLocaleString("en-US")} atoms and ` +
        `<strong>${ca.length}</strong> Cα`,
    );
  }
  if (trajectory !== undefined && topology !== undefined) {
    if (trajectory.particleCount !== topology.atoms.length) {
      // 🔴 Refused rather than analysed. Coordinates are matched to atoms by
      // index, so a topology with a different atom count does not mean "close
      // enough" -- it means every name is attached to the wrong coordinate,
      // and the Cα selection picks 20 arbitrary atoms.
      parts.push(
        '<span class="bad-box">The trajectory has ' +
          `${trajectory.particleCount.toLocaleString("en-US")} atoms and the ` +
          `topology has ${topology.atoms.length.toLocaleString("en-US")}. ` +
          "Coordinates are matched to atoms by index, so these cannot be the " +
          "same system.</span>",
      );
    } else if (caIndices(topology).length < 3) {
      parts.push(
        '<span class="bad-box">Fewer than three Cα atoms, so there is ' +
          "nothing to superpose. Is this a protein?</span>",
      );
    }
  }
  summary.innerHTML =
    parts.length === 0
      ? "Nothing loaded yet."
      : `<p class="note">${parts.join(". ")}.</p>`;
  button.disabled =
    trajectory === undefined ||
    topology === undefined ||
    trajectory.particleCount !== topology.atoms.length ||
    caIndices(topology).length < 3;
}

el<HTMLInputElement>("file-dcd").addEventListener("change", (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file === undefined) return;
  void file.arrayBuffer().then((bytes) => {
    try {
      trajectory = readDcd(new Uint8Array(bytes));
      setStatus(`Loaded ${file.name}.`, "ok");
    } catch (error) {
      trajectory = undefined;
      setStatus(error instanceof Error ? error.message : String(error), "bad");
    }
    describe();
  });
});

el<HTMLInputElement>("file-top").addEventListener("change", (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file === undefined) return;
  void file.text().then((text) => {
    try {
      topology = readPdb(text);
      fillLigandPicker();
      setStatus(`Loaded ${file.name}.`, "ok");
    } catch (error) {
      topology = undefined;
      setStatus(error instanceof Error ? error.message : String(error), "bad");
    }
    describe();
  });
});

const files: { name: string; text?: string; canvas?: string }[] = [];

function offer(): void {
  downloads.innerHTML =
    "<table><tr><th>File</th><th></th></tr>" +
    files
      .map(
        ({ name }) =>
          `<tr><td><code>${escape(name)}</code></td>` +
          `<td><button type="button" data-save="${escape(name)}">Save</button></td></tr>`,
      )
      .join("") +
    "</table>";
  for (const save of downloads.querySelectorAll<HTMLButtonElement>("[data-save]")) {
    save.addEventListener("click", () => {
      const entry = files.find((f) => f.name === save.dataset["save"]);
      if (entry === undefined) return;
      const link = document.createElement("a");
      if (entry.canvas !== undefined) {
        // A PNG straight off the canvas, so the file is what the page shows
        // rather than a second rendering that could differ from it.
        link.href = el<HTMLCanvasElement>(entry.canvas).toDataURL("image/png");
      } else {
        link.href = URL.createObjectURL(
          new Blob([entry.text ?? ""], { type: "text/csv" }),
        );
      }
      link.download = entry.name;
      link.click();
      if (entry.canvas === undefined) URL.revokeObjectURL(link.href);
    });
  }
  window.__mirAnalyse = {
    ...(window.__mirAnalyse ?? { state: "done" }),
    artefacts: files.map(({ name, text }) => ({
      name,
      bytes: text === undefined ? 0 : new Blob([text]).size,
    })),
  };
}

const line = (
  id: string,
  label: string,
  unit: string,
  colour: string,
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
): void => {
  const points = [];
  for (let i = 0; i < ys.length; i++) {
    points.push({ x: xs[i] as number, y: ys[i] as number });
  }
  drawPlot(el<HTMLCanvasElement>(id), { label, unit, colour, points });
};

button.addEventListener("click", () => {
  void (async () => {
    if (trajectory === undefined || topology === undefined) return;
    button.disabled = true;
    files.length = 0;
    window.__mirAnalyse = { state: "analysing" };
    try {
      const report: Record<string, number | string> = {};
      const ca = caIndices(topology);
      const frames = trajectory.frameCount;
      // The time axis, in ps. 🔴 Built from this trajectory's own interval and
      // timestep rather than from the length of another series -- the
      // notebook's cell 40 builds the radius-of-gyration axis from
      // `len(rmsd)`, which is a different length, and the plot is wrong
      // whenever the two series differ (docs/NOTEBOOK_PARITY.md, finding).
      const time = new Float64Array(frames);
      for (let i = 0; i < frames; i++) {
        time[i] = i * trajectory.interval * trajectory.timeStep;
      }

      setStatus("Cα RMSD…");
      await yieldToPaint();
      const rmsd = rmsdSeries(trajectory, ca);
      line("plot-rmsd", "Cα RMSD", "nm", "#6aa9ff", time, rmsd);
      const rmsdKde = kernelDensity(rmsd);
      line("plot-rmsd-kde", "RMSD density", "1/nm", "#6aa9ff", rmsdKde.x, rmsdKde.y);

      setStatus("Radius of gyration…");
      await yieldToPaint();
      const rg = radiusOfGyration(trajectory, ca);
      line("plot-rg", "Radius of gyration", "nm", "#7ddc8a", time, rg);
      const rgKde = kernelDensity(rg);
      line("plot-rg-kde", "Rg density", "1/nm", "#7ddc8a", rgKde.x, rgKde.y);

      setStatus("Cα RMSF…");
      await yieldToPaint();
      const fluctuation = rmsf(trajectory, ca);
      const residues = new Float64Array(ca.length);
      for (const [slot, atom] of ca.entries()) {
        residues[slot] = Number(topology.atoms[atom]?.residueId ?? slot + 1);
      }
      line("plot-rmsf", "Cα RMSF", "nm", "#ffb86a", residues, fluctuation);

      setStatus("2D RMSD…");
      await yieldToPaint();
      const matrix = pairwiseRmsd(trajectory, ca);
      const range = drawHeatmap(el<HTMLCanvasElement>("plot-2d"), matrix, {
        size: frames,
        scale: "sequential",
      });
      el<HTMLParagraphElement>("scale-2d").textContent =
        `0 to ${range.max.toFixed(4)} nm, over ${frames} frames.`;

      setStatus("Principal components…");
      await yieldToPaint();
      const components = pca(trajectory, ca, 2);
      drawPcaScatter(components.projections, frames);
      const total = components.eigenvalues.reduce((sum, v) => sum + Math.max(0, v), 0);
      el<HTMLParagraphElement>("pca-summary").textContent =
        `PC1 carries ${((components.eigenvalues[0]! / total) * 100).toFixed(1)}% of ` +
        `the motion and PC2 ${((components.eigenvalues[1]! / total) * 100).toFixed(1)}%, ` +
        `of ${total.toFixed(4)} nm² in total.`;

      setStatus("Cross-correlation…");
      await yieldToPaint();
      const correl = crossCorrelation(trajectory, ca);
      drawHeatmap(el<HTMLCanvasElement>("plot-correl"), correl, {
        size: ca.length,
        scale: "diverging",
        min: -1,
        max: 1,
      });

      // --- the ligand, when the topology has one -------------------------
      const ligandName = el<HTMLSelectElement>("ligand-residue").value;
      if (ligandName !== "") {
        setStatus("Ligand…");
        await yieldToPaint();
        const ligand = ligandHeavyAtoms(topology, ligandName);
        const cutoff = Number(el<HTMLInputElement>("pocket-cutoff").value) / 10;
        const conformational = ligandRmsd(trajectory, ligand);
        const centre = centreOfMassSeries(trajectory, topology, ligand);
        const travelled = new Float64Array(frames);
        for (let frame = 0; frame < frames; frame++) {
          travelled[frame] = Math.hypot(
            (centre[frame * 3] as number) - (centre[0] as number),
            (centre[frame * 3 + 1] as number) - (centre[1] as number),
            (centre[frame * 3 + 2] as number) - (centre[2] as number),
          );
        }
        line("plot-lig-rmsd", `${ligandName} conformational RMSD`, "nm", "#d98fd9",
          time, conformational);
        line("plot-lig-com", `${ligandName} centre-of-mass displacement`, "nm",
          "#d9b38f", time, travelled);

        const pocket = pocketResidues(
          topology, ligand, trajectory.positions, trajectory.particleCount, 0, cutoff,
        );
        const distances = residueDistances(trajectory, topology, ligand, pocket);
        el<HTMLDivElement>("pocket-body").innerHTML =
          `<p class="note">${pocket.length} residue` +
          `${pocket.length === 1 ? "" : "s"} within ` +
          `${(cutoff * 10).toFixed(1)} Å of ${escape(ligandName)} in frame 0.</p>` +
          "<table><tr><th>Residue</th><th>Mean (nm)</th><th>SD (nm)</th></tr>" +
          distances
            .map(
              (d) =>
                `<tr><td>${d.residue}</td><td>${d.mean.toFixed(4)}</td>` +
                `<td>${d.standardDeviation.toFixed(4)}</td></tr>`,
            )
            .join("") +
          "</table>";

        files.push(
          {
            name: "ligand.csv",
            text: seriesCsv([
              { name: "time_ps", values: time },
              { name: "conformational_rmsd_nm", values: conformational },
              { name: "com_displacement_nm", values: travelled },
            ]),
          },
          {
            name: "pocket_distances.csv",
            text: seriesCsv([
              { name: "time_ps", values: time },
              ...distances.map((d) => ({
                name: `res_${d.residue}_nm`,
                values: d.distances,
              })),
            ]),
          },
          { name: "ligand_rmsd.png", canvas: "plot-lig-rmsd" },
          { name: "ligand_com.png", canvas: "plot-lig-com" },
        );
        report["ligand"] = ligandName;
        report["ligandAtoms"] = ligand.length;
        report["pocketResidues"] = pocket.length;
        report["ligandRmsdMax"] = Number(Math.max(...conformational).toFixed(6));
        report["ligandComMax"] = Number(Math.max(...travelled).toFixed(6));
      }

      files.push(
        {
          name: "analysis.csv",
          text: seriesCsv([
            { name: "time_ps", values: time },
            { name: "rmsd_ca_nm", values: rmsd },
            { name: "radius_of_gyration_nm", values: rg },
            { name: "pc1", values: components.projections.filter((_, i) => i % 2 === 0) },
            { name: "pc2", values: components.projections.filter((_, i) => i % 2 === 1) },
          ]),
        },
        {
          name: "rmsf.csv",
          text: seriesCsv([
            { name: "residue", values: residues },
            { name: "rmsf_nm", values: fluctuation },
          ]),
        },
        { name: "rmsd_2d.csv", text: matrixCsv(matrix, frames) },
        { name: "cross_correlation.csv", text: matrixCsv(correl, ca.length) },
        { name: "rmsd.png", canvas: "plot-rmsd" },
        { name: "radius_of_gyration.png", canvas: "plot-rg" },
        { name: "rmsf.png", canvas: "plot-rmsf" },
        { name: "rmsd_2d.png", canvas: "plot-2d" },
        { name: "pca.png", canvas: "plot-pca" },
        { name: "cross_correlation.png", canvas: "plot-correl" },
      );
      offer();

      Object.assign(report, {
        frames,
        caAtoms: ca.length,
        rmsdMax: Number(Math.max(...rmsd).toFixed(6)),
        rgMean: Number((rg.reduce((a, b) => a + b, 0) / frames).toFixed(6)),
        rmsfMax: Number(Math.max(...fluctuation).toFixed(6)),
        pc1Fraction: Number(((components.eigenvalues[0]! / total) * 100).toFixed(3)),
      });
      window.__mirAnalyse = {
        ...(window.__mirAnalyse ?? { state: "done" }),
        state: "done",
        summary: report,
      };
      setStatus(`Analysed ${frames} frames over ${ca.length} Cα atoms.`, "ok");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      window.__mirAnalyse = { state: "failed", error: message };
      setStatus(message, "bad");
    } finally {
      describe();
    }
  })();
});

/**
 * PC1 against PC2, each point coloured by its frame index.
 *
 * Drawn here rather than through `plot.ts` because that draws a line through
 * ordered points, and a PCA scatter is the one plot where the *order* must not
 * be joined up: consecutive frames can sit far apart, and a line between them
 * suggests a path through conformational space that the sampling does not
 * support. Colour carries the time instead.
 */
function drawPcaScatter(projections: Float64Array, frames: number): void {
  const canvas = el<HTMLCanvasElement>("plot-pca");
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(canvas.clientWidth * ratio);
  canvas.height = Math.round(canvas.clientHeight * ratio);
  const context = canvas.getContext("2d");
  if (context === null) return;
  context.clearRect(0, 0, canvas.width, canvas.height);

  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < frames; i++) {
    const x = projections[i * 2] as number;
    const y = projections[i * 2 + 1] as number;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  const pad = 24 * ratio;
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const toX = (x: number): number =>
    pad + ((x - minX) / spanX) * (canvas.width - 2 * pad);
  const toY = (y: number): number =>
    canvas.height - pad - ((y - minY) / spanY) * (canvas.height - 2 * pad);

  context.strokeStyle = "rgba(128,128,128,0.35)";
  context.lineWidth = ratio;
  context.beginPath();
  context.moveTo(toX(0), pad);
  context.lineTo(toX(0), canvas.height - pad);
  context.moveTo(pad, toY(0));
  context.lineTo(canvas.width - pad, toY(0));
  context.stroke();

  for (let i = 0; i < frames; i++) {
    const t = frames === 1 ? 1 : i / (frames - 1);
    // Dark early, light late -- the notebook's time colouring.
    context.fillStyle = `hsl(${220 - 180 * t}, 70%, ${28 + 42 * t}%)`;
    context.beginPath();
    context.arc(
      toX(projections[i * 2] as number),
      toY(projections[i * 2 + 1] as number),
      3 * ratio,
      0,
      Math.PI * 2,
    );
    context.fill();
  }

  context.fillStyle = "rgba(160,160,160,0.9)";
  context.font = `${11 * ratio}px system-ui, sans-serif`;
  context.fillText("PC1 →", canvas.width - 54 * ratio, canvas.height - 8 * ratio);
  context.save();
  context.translate(10 * ratio, 46 * ratio);
  context.rotate(-Math.PI / 2);
  context.fillText("PC2 →", 0, 0);
  context.restore();
}

/**
 * Cell 33: one frame as a structure, solvent dropped.
 *
 * Separate from the Analyse button because it is a different question — "give
 * me this frame" rather than "measure the trajectory" — and because a user
 * picking through frames should not re-run a PCA each time.
 */
el<HTMLButtonElement>("extract").addEventListener("click", () => {
  const note = el<HTMLSpanElement>("extract-note");
  if (trajectory === undefined || topology === undefined) {
    note.textContent = "Load a trajectory and a topology first.";
    return;
  }
  const index = Number(el<HTMLInputElement>("extract-frame").value);
  if (!Number.isInteger(index) || index < 0 || index >= trajectory.frameCount) {
    note.textContent = `Frame must be between 0 and ${trajectory.frameCount - 1}.`;
    return;
  }
  const frame = extractFrame(topology, trajectory, index);
  const text = writePdb(frame, {
    remark: `FRAME ${index} OF ${trajectory.frameCount}, SOLVENT REMOVED`,
  });
  const url = URL.createObjectURL(new Blob([text], { type: "chemical/x-pdb" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `frame_${index}.pdb`;
  link.click();
  URL.revokeObjectURL(url);
  note.textContent =
    `${frame.atoms.length.toLocaleString("en-US")} atoms, ` +
    `${frame.residues.length} residues, solvent removed.`;
});

window.__mirAnalyse = { state: "idle" };
fillLigandPicker();
describe();
