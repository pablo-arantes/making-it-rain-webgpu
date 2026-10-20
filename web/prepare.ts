/**
 * The preparation page: a deposited structure in, an OpenMM `System` out.
 *
 * This is the path `src/prep/` exists for, wired to something a person can
 * use. The pipeline itself is tested headlessly against OpenMM term by term
 * (`test/prep-*.test.ts`); what this file adds is the parts that only matter
 * with a user in front of them.
 *
 * Three of those are worth naming.
 *
 * **The force field files are fetched, not bundled.** Together they are about
 * 1.5 MB — `protein.ff14SB.xml` alone is 224 kB and the water box is 180 kB —
 * and none of it is needed to look at the page. They load on demand and are
 * cached for the session, so the first `Prepare` is slower than the second and
 * nothing is paid by someone who only wanted the run page.
 *
 * **Deviations are shown, not filed.** `AGENTS.md` requires that the physics
 * never change silently, and a deviation recorded only in a document in the
 * repository is silent to the person running the simulation. Section 4 lists
 * the ones that actually applied to what was just built — the ion placement is
 * only listed when ions were added — with the numbers that came out of this
 * run rather than generic text.
 *
 * **Preparation runs on the main thread and will block it.** Solvating a large
 * protein is seconds of tight loops. The status line is updated and the frame
 * yielded before each stage, so the page says what it is doing rather than
 * going white; a worker is the real answer and is not written.
 */

import { writePdb, readPdb, type PdbTopology } from "../src/analysis/pdb.js";
import { serialiseState } from "../src/engine/state.js";
import {
  NonbondedMethod,
  serialiseSystem,
  type OpenMMSystem,
} from "../src/engine/system.js";
import { totalCharge } from "../src/prep/assign.js";
import {
  inferBonds,
  parseStandardBonds,
  type StandardBondTable,
} from "../src/prep/bonds.js";
import {
  mergeForceFields,
  parseForceField,
  UnsupportedForceField,
  type ForceField,
} from "../src/prep/ffxml.js";
import {
  addHydrogens,
  isHydrogenAtom,
  isWater,
  parseHydrogenDefinitions,
  stripAtoms,
  type HydrogenDefinitions,
} from "../src/prep/hydrogens.js";
import { prepareSystem } from "../src/prep/prepare.js";
import {
  boundingSphereRadius,
  solvate,
  type SolvateResult,
} from "../src/prep/solvate.js";
import { buildSystem } from "../src/prep/system.js";
import { AmbiguousTemplate, TemplateMatchFailed } from "../src/prep/template.js";

declare global {
  interface Window {
    /** The Playwright lane reads this; nothing in the page depends on it. */
    __mirPrepare?: {
      state: "idle" | "loading" | "preparing" | "done" | "failed";
      error?: string;
      report?: Record<string, number | string>;
      artefacts?: readonly { name: string; bytes: number }[];
    };
  }
}

function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) throw new Error(`no element #${id}`);
  return found as T;
}

const status = el<HTMLParagraphElement>("prepare-status");
const structureSummary = el<HTMLDivElement>("structure-summary");
const report = el<HTMLDivElement>("report");
const deviations = el<HTMLDivElement>("deviations");
const downloads = el<HTMLDivElement>("downloads");
const prepareButton = el<HTMLButtonElement>("prepare");

function setStatus(text: string, kind: "" | "ok" | "bad" = ""): void {
  status.textContent = text;
  status.className = kind === "" ? "status" : `status ${kind}`;
}

/** Let the browser paint before a long synchronous stage. */
const yieldToPaint = (): Promise<void> =>
  new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));

const escape = (value: string): string =>
  value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// --- The data files, fetched once ---------------------------------------
//
// Imported with `?url` rather than `?raw`, so each one is a separate request
// the browser can cache instead of 1.5 MB inlined into the page's JavaScript.
import residuesUrl from "../vendor/ffxml/residues.xml?url";
import hydrogensUrl from "../vendor/ffxml/hydrogens.xml?url";
import ff14sbUrl from "../vendor/ffxml/protein.ff14SB.xml?url";
import ildnUrl from "../vendor/ffxml/amber99sbildn.xml?url";
import sbUrl from "../vendor/ffxml/amber99sb.xml?url";
import tip3pXmlUrl from "../vendor/ffxml/amber14-tip3p.xml?url";
import spceXmlUrl from "../vendor/ffxml/amber14-spce.xml?url";
import tip3pBoxUrl from "../vendor/water/tip3p.pdb?url";
import spceBoxUrl from "../vendor/water/spce.pdb?url";

const PROTEIN_URLS: Readonly<Record<string, string>> = {
  "protein.ff14SB.xml": ff14sbUrl,
  "amber99sbildn.xml": ildnUrl,
  "amber99sb.xml": sbUrl,
};
const WATER_XML_URLS: Readonly<Record<string, string>> = {
  tip3p: tip3pXmlUrl,
  spce: spceXmlUrl,
};
const WATER_BOX_URLS: Readonly<Record<string, string>> = {
  tip3p: tip3pBoxUrl,
  spce: spceBoxUrl,
};

const cache = new Map<string, Promise<string>>();
function fetchText(url: string): Promise<string> {
  const existing = cache.get(url);
  if (existing !== undefined) return existing;
  const pending = fetch(url).then(async (response) => {
    if (!response.ok) {
      throw new Error(`could not load ${url}: HTTP ${response.status}`);
    }
    return response.text();
  });
  cache.set(url, pending);
  return pending;
}

// --- Loaded structure ---------------------------------------------------

let loaded: { name: string; text: string; topology: PdbTopology } | undefined;

function describeStructure(): void {
  if (loaded === undefined) {
    structureSummary.textContent = "Nothing loaded yet.";
    prepareButton.disabled = true;
    return;
  }
  const { topology } = loaded;
  const hydrogens = topology.atoms.filter((a) => a.element.toUpperCase() === "H").length;
  const waters = topology.residues.filter((r) => r.name === "HOH" || r.name === "WAT").length;
  const hetatms = new Set(
    topology.atoms
      .filter((a) => a.hetatm && a.residueName !== "HOH")
      .map((a) => a.residueName),
  );
  structureSummary.innerHTML =
    `<p class="note"><strong>${escape(loaded.name)}</strong>: ` +
    `${topology.atoms.length.toLocaleString("en-US")} atoms, ` +
    `${topology.residues.length.toLocaleString("en-US")} residues, ` +
    `${topology.chains.length} chain${topology.chains.length === 1 ? "" : "s"}. ` +
    `${hydrogens} hydrogen${hydrogens === 1 ? "" : "s"} and ${waters} water` +
    `${waters === 1 ? "" : "s"} will be removed and rebuilt.` +
    (hetatms.size === 0
      ? ""
      : ` <strong>Ignored:</strong> ${escape([...hetatms].join(", "))} — ` +
        "this page prepares protein in water, and a heterogen needs the " +
        "ligand path, which is Phase 4.") +
    "</p>";
  prepareButton.disabled = false;
}

function load(name: string, text: string): void {
  try {
    const topology = readPdb(text);
    if (topology.atoms.length === 0) {
      throw new Error("the file has no ATOM or HETATM records");
    }
    loaded = { name, text, topology };
    describeStructure();
    setStatus(`Loaded ${name}.`, "ok");
  } catch (error) {
    loaded = undefined;
    describeStructure();
    setStatus(error instanceof Error ? error.message : String(error), "bad");
  }
}

el<HTMLInputElement>("file-pdb").addEventListener("change", (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file === undefined) return;
  void file.text().then((text) => load(file.name, text));
});

el<HTMLButtonElement>("fetch").addEventListener("click", () => {
  const id = el<HTMLInputElement>("pdb-id").value.trim();
  if (!/^[0-9A-Za-z]{4}$/.test(id)) {
    setStatus("A PDB ID is four characters, like 1L2Y.", "bad");
    return;
  }
  // 🔴 This asks the RCSB for the structure, which means the ID leaves the
  // tab. Uploading a file does not. The lede says so, because "your data
  // never leaves the browser" would be false for this button.
  setStatus(`Asking the RCSB for ${id.toUpperCase()}…`);
  void fetch(`https://files.rcsb.org/download/${id.toUpperCase()}.pdb`)
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(
          response.status === 404
            ? `The RCSB has no entry ${id.toUpperCase()}.`
            : `The RCSB returned HTTP ${response.status}.`,
        );
      }
      load(`${id.toUpperCase()}.pdb`, await response.text());
    })
    .catch((error: unknown) => {
      setStatus(
        `Could not fetch ${id.toUpperCase()}: ` +
          `${error instanceof Error ? error.message : String(error)}. ` +
          "Downloading the file and uploading it works too.",
        "bad",
      );
    });
});

// --- Downloads ----------------------------------------------------------

function offer(files: readonly { name: string; text: string }[]): void {
  downloads.innerHTML =
    "<table><tr><th>File</th><th>Size</th><th></th></tr>" +
    files
      .map(({ name, text }) => {
        const bytes = new Blob([text]).size;
        return (
          `<tr><td><code>${escape(name)}</code></td>` +
          `<td>${(bytes / 1024).toFixed(1)} kB</td>` +
          `<td><button type="button" data-download="${escape(name)}">Save</button></td></tr>`
        );
      })
      .join("") +
    "</table>";
  for (const button of downloads.querySelectorAll<HTMLButtonElement>("[data-download]")) {
    button.addEventListener("click", () => {
      const wanted = button.dataset["download"];
      const file = files.find((f) => f.name === wanted);
      if (file === undefined) return;
      const url = URL.createObjectURL(new Blob([file.text], { type: "text/plain" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = file.name;
      link.click();
      URL.revokeObjectURL(url);
    });
  }
  window.__mirPrepare = {
    ...(window.__mirPrepare ?? { state: "done" }),
    artefacts: files.map(({ name, text }) => ({
      name,
      bytes: new Blob([text]).size,
    })),
  };
}

/**
 * The deviations that applied to this run, with this run's numbers.
 *
 * Deliberately not the whole table from `docs/DEVIATIONS.md`: a list that
 * includes things which did not happen trains people to skip it. Each entry
 * names the number it is about.
 */
function showDeviations(solvated: SolvateResult, variants: readonly (string | null)[]): void {
  const entries: string[] = [];

  entries.push(
    "<li><strong>Hydrogen positions are constructed, not minimised</strong> " +
      "(deviation 16). OpenMM offsets each hydrogen from its parent using a " +
      "random vector and then minimises them; it does not reproduce its own " +
      "hydrogen positions between runs, so there is nothing to match. Which " +
      "hydrogens were added, and their names, do match atom for atom. " +
      "<em>Minimise before running — the run page does this first.</em></li>",
  );

  const histidines = variants.filter((v) => v === "HID" || v === "HIE" || v === "HIP");
  if (histidines.length > 0) {
    entries.push(
      `<li><strong>${histidines.length} histidine` +
        `${histidines.length === 1 ? "" : "s"} set to ` +
        `${escape([...new Set(histidines)].join(", "))}</strong> ` +
        "(deviation 17). OpenMM documents a choice between HID and HIE based " +
        "on hydrogen bonding, but that code cannot be reached — it always " +
        "returns HID — so this matches OpenMM rather than the documentation. " +
        "The two have different charges.</li>",
    );
  }

  const ions = solvated.positiveIons + solvated.negativeIons;
  if (ions > 0) {
    entries.push(
      `<li><strong>${ions} ion${ions === 1 ? "" : "s"} placed from the seed</strong> ` +
        "(deviation 18). OpenMM picks which waters to replace at random and " +
        "cannot repeat it; this uses the seed above, so the same seed gives " +
        "the same box. The counts are OpenMM's exactly.</li>",
    );
  }

  entries.push(
    "<li><strong>The box is a cube</strong> (deviation 19). A rhombic " +
      "dodecahedron would hold about 70% of the water for the same padding, " +
      "and needs a triclinic minimum image that is not implemented. It is " +
      "refused rather than quietly substituted.</li>",
  );

  deviations.innerHTML = `<ul class="note">${entries.join("")}</ul>`;
}

// --- Prepare ------------------------------------------------------------

async function loadForceField(
  protein: string,
  water: string,
): Promise<{ field: ForceField; bonds: StandardBondTable; hydrogens: HydrogenDefinitions; box: PdbTopology }> {
  const [proteinXml, waterXml, residuesXml, hydrogensXml, boxPdb] = await Promise.all([
    fetchText(PROTEIN_URLS[protein] ?? ff14sbUrl),
    fetchText(WATER_XML_URLS[water] ?? tip3pXmlUrl),
    fetchText(residuesUrl),
    fetchText(hydrogensUrl),
    fetchText(WATER_BOX_URLS[water] ?? tip3pBoxUrl),
  ]);
  return {
    field: mergeForceFields([parseForceField(proteinXml), parseForceField(waterXml)]),
    bonds: parseStandardBonds(residuesXml),
    hydrogens: parseHydrogenDefinitions(hydrogensXml),
    box: readPdb(boxPdb),
  };
}

prepareButton.addEventListener("click", () => {
  void (async () => {
    if (loaded === undefined) return;
    prepareButton.disabled = true;
    window.__mirPrepare = { state: "loading" };
    try {
      const protein = el<HTMLSelectElement>("forcefield").value;
      const water = el<HTMLSelectElement>("water").value;
      const ph = Number(el<HTMLInputElement>("ph").value);
      const padding = Number(el<HTMLInputElement>("padding").value) / 10;
      const concentration = Number(el<HTMLInputElement>("concentration").value);
      const positiveIon = el<HTMLSelectElement>("positive-ion").value;
      const seed = Number(el<HTMLInputElement>("prep-seed").value);

      setStatus("Loading the force field…");
      await yieldToPaint();
      const data = await loadForceField(protein, water);

      window.__mirPrepare = { state: "preparing" };
      setStatus("Removing water and hydrogens…");
      await yieldToPaint();
      const heavy = stripAtoms(
        stripAtoms(loaded.topology, isWater).topology,
        isHydrogenAtom,
      ).topology;

      setStatus("Adding hydrogens…");
      await yieldToPaint();
      const hydrogenated = addHydrogens(
        heavy,
        inferBonds(heavy, data.bonds),
        data.hydrogens,
        ph,
      );

      setStatus("Assigning parameters to the solute…");
      await yieldToPaint();
      const solute = prepareSystem(hydrogenated.topology, data.field, data.bonds);

      setStatus("Filling the box with water…");
      await yieldToPaint();
      const solvated = solvate(
        hydrogenated.topology,
        hydrogenated.bonds,
        solute.assignment.atoms,
        data.box,
        {
          padding,
          model: water,
          ionicStrength: concentration,
          neutralize: true,
          positiveIon,
          negativeIon: "Cl-",
          seed,
        },
      );

      setStatus("Assigning parameters to the whole system…");
      await yieldToPaint();
      const prepared = prepareSystem(solvated.topology, data.field, data.bonds);
      const system: OpenMMSystem = buildSystem(prepared, {
        nonbondedMethod: NonbondedMethod.PME,
        cutoff: 1,
        constraints: "h-bonds",
        removeCMMotion: true,
      });

      const charge = totalCharge(prepared.assignment);

      // 🔴 The solute's bounding-sphere radius, reported because it is what
      // sets the box.
      //
      // An earlier version of this reported "padding achieved" and warned when
      // it fell short. That metric was vacuous: `addSolvent` *defines* the box
      // as 2 * radius + padding, so the padding delivered always equals the
      // padding asked for and the warning could never fire. A browser test
      // caught it.
      //
      // The real difference from OpenMM is the radius, not the padding. This
      // page places hydrogens geometrically instead of minimising them
      // (deviation 16), so the outermost atom sits a little closer in: on 1L2Y
      // at 4 Å the radius is 1.3494 nm against OpenMM's 1.3882, giving a
      // 3.0988 nm box where OpenMM gives 3.1763 and 697 waters where it gives
      // 694. Each box satisfies the requested padding for its own structure,
      // so this changes the water count and the cost of a run, not the
      // minimum-image guarantee.
      const radius = boundingSphereRadius(
        hydrogenated.topology.positions,
        hydrogenated.topology.atoms.length,
      );

      const fields: Record<string, number | string> = {
        "Solute atoms": hydrogenated.topology.atoms.length,
        "Hydrogens added": hydrogenated.added.length,
        Waters: solvated.waters,
        [`${positiveIon} ions`]: solvated.positiveIons,
        "Cl- ions": solvated.negativeIons,
        "Total atoms": solvated.topology.atoms.length,
        "Box (nm)": solvated.boxVectors.map((v) => v.toFixed(4)).join(" × "),
        "Padding (Å)": (padding * 10).toFixed(1),
        "Solute radius (nm)": radius.toFixed(4),
        "Solute charge (e)": solvated.soluteCharge,
        "Total charge (e)": charge.toFixed(6),
        Constraints: system.constraintCount,
        Bonds: prepared.bondTerms.length,
        Angles: prepared.angleTerms.length,
        Torsions: prepared.torsionTerms.length,
        Exceptions: prepared.exceptions.length,
      };
      report.innerHTML =
        "<table>" +
        Object.entries(fields)
          .map(
            ([name, value]) =>
              `<tr><td>${escape(name)}</td><td><strong>${escape(
                typeof value === "number" ? value.toLocaleString("en-US") : value,
              )}</strong></td></tr>`,
          )
          .join("") +
        "</table>" +
        (Math.abs(charge) > 1e-6
          ? '<p class="bad-box">The system is not neutral. That is a bug — ' +
            "neutralisation is on — and it should be reported rather than run.</p>"
          : '<p class="note">Neutral, as it should be.</p>');

      showDeviations(solvated, hydrogenated.variants);

      const remark = `PREPARED IN A BROWSER, ${protein} ${water} pH ${ph}`;
      offer([
        { name: "system.xml", text: serialiseSystem(system) },
        {
          name: "state.xml",
          text: serialiseState({
            openmmVersion: "8.4",
            stepCount: 0,
            time: 0,
            boxVectors: [
              [solvated.boxVectors[0], 0, 0],
              [0, solvated.boxVectors[1], 0],
              [0, 0, solvated.boxVectors[2]],
            ],
            particleCount: solvated.topology.atoms.length,
            positions: solvated.topology.positions,
          }),
        },
        { name: "system.pdb", text: writePdb(solvated.topology, { remark }) },
      ]);

      window.__mirPrepare = {
        ...(window.__mirPrepare ?? { state: "done" }),
        state: "done",
        report: fields,
      };
      setStatus(
        `Prepared: ${solvated.topology.atoms.length.toLocaleString("en-US")} atoms. ` +
          "Save the files, then open the run page.",
        "ok",
      );
    } catch (error) {
      const message =
        error instanceof TemplateMatchFailed ||
        error instanceof AmbiguousTemplate ||
        error instanceof UnsupportedForceField
          ? error.message
          : error instanceof Error
            ? error.message
            : String(error);
      report.innerHTML = `<p class="bad-box">${escape(message)}</p>`;
      window.__mirPrepare = { state: "failed", error: message };
      setStatus("Preparation failed; the panel says why.", "bad");
    } finally {
      prepareButton.disabled = loaded === undefined;
    }
  })();
});

window.__mirPrepare = { state: "idle" };
describeStructure();
