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
import { type BoxVectors, isRectangular } from "../src/reference/triclinic.js";
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
  isWater,
  parseHydrogenDefinitions,
  rebuildableHydrogen,
  stripAtoms,
  type HydrogenDefinitions,
} from "../src/prep/hydrogens.js";
import { findLigandCandidates } from "../src/prep/ligand.js";
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
/** An uploaded ligand ffxml, parsed so a bad file is reported on upload. */
let ligand: { name: string; field: ForceField; templates: readonly string[] } | undefined;

const ligandSummary = el<HTMLDivElement>("ligand-summary");

function describeLigand(): void {
  if (ligand === undefined) {
    ligandSummary.innerHTML =
      '<p class="note">No ligand parameters. Heterogens will be dropped.</p>';
  } else {
    ligandSummary.innerHTML =
      `<p class="note"><strong>${escape(ligand.name)}</strong>: ` +
      `${ligand.templates.length} residue template` +
      `${ligand.templates.length === 1 ? "" : "s"}, ` +
      `${ligand.field.types.size} atom types. ` +
      // 🔴 Named, because the template name is almost never the residue name:
      // openmmforcefields names it after the molecule's SMILES. Matching is by
      // graph, so that is fine -- but a user comparing the two would otherwise
      // think the file was wrong.
      `Template${ligand.templates.length === 1 ? "" : "s"} named ` +
      `<code>${escape(ligand.templates.map((t) => t.slice(0, 40)).join(", "))}</code>, ` +
      "which need not match your residue name: the ligand is matched by the " +
      "shape of its bonds.</p>";
  }
  describeStructure();
}

el<HTMLInputElement>("file-ligand").addEventListener("change", (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file === undefined) {
    ligand = undefined;
    describeLigand();
    return;
  }
  void file.text().then((text) => {
    try {
      const field = parseForceField(text);
      if (field.templates.length === 0) {
        throw new Error(
          "this file has no residue template, so it cannot describe a ligand. " +
            "An ffxml from openmmforcefields' `generate_residue_template` has one.",
        );
      }
      ligand = {
        name: file.name,
        field,
        templates: field.templates.map((t) => t.name),
      };
      setStatus(`Loaded ligand parameters from ${file.name}.`, "ok");
    } catch (error) {
      ligand = undefined;
      setStatus(
        `${file.name} is not usable ligand parameters: ` +
          `${error instanceof Error ? error.message : String(error)}`,
        "bad",
      );
    }
    describeLigand();
  });
});

/**
 * The heterogen picker.
 *
 * 🔴 Asking rather than inferring, and the reason is a specific trap. The
 * notebook renames every ligand to `UNK`, and OpenMM's `hydrogens.xml` *has* a
 * `UNK` entry — an unknown **amino acid**, with hydrogens on N, CA, CB and CG.
 * So a ligand called `UNK` looks to this pipeline like a residue whose
 * hydrogens can be rebuilt: they get stripped, and then nothing is put back,
 * because the ligand has no backbone for those definitions to attach to. JZ4
 * arrived as 10 atoms against its template's 22 and matched nothing.
 *
 * Naming the ligand residue explicitly removes the guess. The chosen residue is
 * excluded from hydrogen stripping and from heterogen dropping; everything else
 * non-polymer is dropped.
 */
function fillLigandPicker(): void {
  const select = el<HTMLSelectElement>("ligand-residue");
  const previous = select.value;
  select.innerHTML = '<option value="">(none — drop all heterogens)</option>';
  if (loaded === undefined) return;
  // Grouped by residue *name*, because that is what the pipeline keys on:
  // listing each of fifty chlorides separately would be fifty options that all
  // mean the same thing.
  const groups = new Map<
    string,
    { copies: number; heavy: number; hydrogens: number; additive: boolean }
  >();
  for (const candidate of findLigandCandidates(loaded.topology)) {
    if (candidate.code === "HOH" || candidate.code === "WAT") continue;
    const existing = groups.get(candidate.code);
    if (existing === undefined) {
      groups.set(candidate.code, {
        copies: 1,
        heavy: candidate.heavyAtoms,
        hydrogens: candidate.hydrogens,
        additive: candidate.likelyAdditive,
      });
    } else {
      existing.copies++;
    }
  }
  for (const [code, group] of [...groups].sort(
    (a, b) => Number(a[1].additive) - Number(b[1].additive) || b[1].heavy - a[1].heavy,
  )) {
    const option = document.createElement("option");
    option.value = code;
    option.textContent =
      `${code} — ${group.heavy} heavy atom${group.heavy === 1 ? "" : "s"}` +
      `${group.hydrogens > 0 ? `, ${group.hydrogens} H` : ", no hydrogens"}` +
      `${group.copies > 1 ? `, ${group.copies} copies` : ""}` +
      `${group.additive ? " (looks like an additive)" : ""}`;
    select.append(option);
  }
  // The largest non-additive candidate is first, so it is the useful default.
  const first = [...select.options].find((o) => o.value !== "" && !o.textContent?.includes("additive"));
  select.value =
    [...select.options].some((o) => o.value === previous) && previous !== ""
      ? previous
      : (first?.value ?? "");
}

const chosenLigand = (): string => el<HTMLSelectElement>("ligand-residue").value;

el<HTMLSelectElement>("ligand-residue").addEventListener("change", () => {
  describeStructure();
  describeLigand();
});

function describeStructure(): void {
  if (loaded === undefined) {
    structureSummary.textContent = "Nothing loaded yet.";
    prepareButton.disabled = true;
    return;
  }
  const { topology } = loaded;
  const hydrogens = topology.atoms.filter((a) => a.element.toUpperCase() === "H").length;
  const waters = topology.residues.filter((r) => r.name === "HOH" || r.name === "WAT").length;
  // 🔴 By residue name, not by the HETATM flag. A ligand is often written as
  // `ATOM` by a modelling program, and a flag-based test then reports no
  // heterogens on a structure that has one.
  const hetatms = new Set(
    findLigandCandidates(topology)
      .filter((c) => c.code !== "HOH" && c.code !== "WAT")
      .map((c) => c.code),
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
      : ligand === undefined
      ? ` <strong>Dropped:</strong> ${escape([...hetatms].join(", "))} — ` +
        "no ligand parameters are loaded, so these are removed. Upload an " +
        "ffxml above to keep one."
      : ` <strong>Kept:</strong> ${escape([...hetatms].join(", "))}, using the ` +
        "uploaded parameters. Its hydrogens are taken from your file as they " +
        "are, because nothing here can rebuild them.") +
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
    fillLigandPicker();
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
/** `x × y × z` for a rectangular cell; all three vectors for a skewed one. */
function formatBox(box: BoxVectors): string {
  if (isRectangular(box)) {
    return box.map((v, axis) => (v[axis] as number).toFixed(4)).join(" × ");
  }
  return box.map((v) => `(${v.map((c) => c.toFixed(4)).join(", ")})`).join(" ");
}

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
    // 🔴 The protein file comes first and the ligand last, and the order is
    // load-bearing: `mergeForceFields` keeps the *first* file's 1-4 scale
    // factors, matching OpenMM. A SMIRNOFF ligand ffxml writes
    // `coulomb14scale="0.8333333333"` where ff14SB writes
    // `0.8333333333333334`, and letting the ligand win would apply its coarser
    // value to every protein 1-4 pair.
    field: mergeForceFields([
      parseForceField(proteinXml),
      parseForceField(waterXml),
      ...(ligand === undefined ? [] : [ligand.field]),
    ]),
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
      // 🔴 Only hydrogens on residues the definitions cover. A ligand's cannot
      // be rebuilt -- `hydrogens.xml` has no entry for it -- so removing them
      // would leave a fragment matching no template: JZ4 would arrive as 10
      // atoms against its ffxml's 22.
      const keepResidue = chosenLigand().toUpperCase();
      if (keepResidue !== "" && ligand === undefined) {
        throw new Error(
          `${keepResidue} is selected as the ligand but no parameters are ` +
            "loaded for it. Upload an ffxml, or set the ligand to (none) to " +
            "drop it.",
        );
      }
      const withoutWater = stripAtoms(loaded.topology, isWater).topology;

      // Everything non-polymer except the chosen ligand goes: it has no
      // parameters, and dropping it here gives a clearer failure than letting
      // template matching meet it later.
      const polymerOnly = stripAtoms(withoutWater, (atom) => {
        const name = atom.residueName.toUpperCase();
        if (name === keepResidue) return false;
        return !data.hydrogens.has(name) && !isWater(atom);
      }).topology;

      // 🔴 The chosen ligand keeps its hydrogens whatever `hydrogens.xml` says
      // about its name. See `fillLigandPicker`: a ligand called `UNK` collides
      // with that file's unknown-amino-acid entry, and the collision costs all
      // twelve of JZ4's hydrogens.
      const rebuildable = rebuildableHydrogen(data.hydrogens);
      const heavy = stripAtoms(polymerOnly, (atom) =>
        atom.residueName.toUpperCase() === keepResidue ? false : rebuildable(atom),
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

      const ligandAtoms =
        keepResidue === ""
          ? 0
          : hydrogenated.topology.residues
              .filter((r) => r.name.toUpperCase() === keepResidue)
              .reduce((n, r) => n + r.atoms.length, 0);

      const fields: Record<string, number | string> = {
        "Solute atoms": hydrogenated.topology.atoms.length,
        ...(keepResidue === ""
          ? {}
          : { [`Ligand ${keepResidue} atoms`]: ligandAtoms }),
        "Hydrogens added": hydrogenated.added.length,
        Waters: solvated.waters,
        [`${positiveIon} ions`]: solvated.positiveIons,
        "Cl- ions": solvated.negativeIons,
        "Total atoms": solvated.topology.atoms.length,
        "Box (nm)": formatBox(solvated.boxVectors),
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
            // 🔴 The vectors as solvation produced them. This used to rebuild
            // a diagonal from them, which was right only while a cube was the
            // only shape -- for a dodecahedron it writes a rectangular cell of
            // the correct volume and the wrong shape, and the engine would
            // have run it without complaint.
            boxVectors: solvated.boxVectors,
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
