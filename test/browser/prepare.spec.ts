/**
 * The preparation page, driven as a user drives it.
 *
 * The pipeline underneath is tested term by term against OpenMM in Node
 * (`test/prep-*.test.ts`), so what only this can check is the wiring: that a
 * file arriving through a real `<input type="file">` reaches the reader, that
 * the force field files load over HTTP rather than only from disk, that
 * pressing Prepare produces a system whose numbers match what the headless
 * tests measured, and that the artefacts come out as saveable text.
 *
 * It also checks the two things that are only true in a browser: that the
 * deviations are *shown* rather than filed, which `AGENTS.md` requires, and
 * that the `system.xml` the page offers parses back to the same system — the
 * user's artefact is the whole point of preparing in a tab.
 */

import { expect, test } from "@playwright/test";
import { existsSync, readFileSync } from "node:fs";
import { parseSystem } from "../../src/engine/system.js";

const receptor = "fixtures/1L2Y/receptor.pdb";
const present = existsSync(receptor);

test.describe("the prepare page", () => {
  test.skip(!present, `${receptor} not generated — see tools/README.md`);

  test("is reachable from the front page", async ({ page }) => {
    // The page existed for a while before anything linked to it, which made it
    // reachable only by typing the URL. A link is the difference between
    // shipped and merely built.
    await page.goto("/");
    const link = page.locator('nav a[href="./prepare.html"]');
    await expect(link).toBeVisible();
    await link.click();
    await expect(page.locator("h1")).toHaveText("Prepare a system");
    await page.waitForFunction(() => window.__mirPrepare?.state === "idle");
  });

  test("prepares a deposited structure into a runnable system", async ({ page }) => {
    const failures: string[] = [];
    page.on("pageerror", (error) => failures.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") failures.push(message.text());
    });

    await page.goto("/prepare.html");
    await page.waitForFunction(() => window.__mirPrepare?.state === "idle");

    // --- Load, through the real file input -------------------------------
    await page.locator("#file-pdb").setInputFiles(receptor);
    await expect(page.locator("#structure-summary")).toContainText("atoms");
    await expect(page.locator("#prepare")).toBeEnabled();
    // The deposited 1L2Y is an NMR structure and carries hydrogens, which this
    // page replaces. The summary has to say so, or a user cannot tell why the
    // atom count changes.
    await expect(page.locator("#structure-summary")).toContainText("rebuilt");

    // --- Prepare, at the notebook's defaults ------------------------------
    await page.locator("#prepare").click();
    await page.waitForFunction(
      () =>
        window.__mirPrepare?.state === "done" ||
        window.__mirPrepare?.state === "failed",
      null,
      { timeout: 180_000 },
    );
    const result = await page.evaluate(() => window.__mirPrepare);
    expect(result?.error ?? "").toBe("");
    expect(result?.state).toBe("done");

    const report = result?.report ?? {};

    // The hydrogen *count* is exact against OpenMM -- 154 heavy atoms become
    // 304 -- because which hydrogens are added is parity-tested. Their
    // positions are not (deviation 16), and that propagates: the box is sized
    // from the solute's bounding sphere, so a hydrogen 0.04 nm nearer the
    // centre than OpenMM's minimised one gives a smaller box and a different
    // number of waters. Measured: 3.0988 nm and 697 waters here, against
    // 3.1763 nm and 694 from OpenMM's own hydrogens. So these are checked as
    // "close", with the gap asserted to stay small, rather than as equal --
    // asserting 694 here would be asserting that a documented deviation does
    // not exist.
    expect(report["Solute atoms"]).toBe(304);
    expect(report["Hydrogens added"]).toBe(150);
    expect(Number(report["Waters"])).toBeGreaterThan(680);
    expect(Number(report["Waters"])).toBeLessThan(710);
    // The ion counts come out of the water count and still land on OpenMM's.
    expect(report["Na+ ions"]).toBe(2);
    expect(report["Cl- ions"]).toBe(3);
    // Internally consistent: solute + 3 per water + one atom per ion.
    expect(report["Total atoms"]).toBe(304 + 3 * Number(report["Waters"]) + 5);
    expect(Number(String(report["Box (nm)"]).split(" ")[0])).toBeCloseTo(3.1, 1);
    // Neutral: +1 peptide, 2 Na+, 3 Cl-.
    expect(Number(report["Total charge (e)"])).toBeCloseTo(0, 6);
    await expect(page.locator("#report")).toContainText("Neutral");

    // The box is derived from the solute's bounding sphere, so the radius is
    // reported next to the padding -- that radius is where the difference from
    // OpenMM lives, since the padding itself is satisfied by construction.
    expect(Number(report["Padding (Å)"])).toBeCloseTo(4, 6);
    expect(Number(report["Solute radius (nm)"])).toBeGreaterThan(1.3);
    expect(Number(report["Solute radius (nm)"])).toBeLessThan(1.45);
    // 2 * radius + padding, which is how addSolvent sizes a cube.
    expect(Number(String(report["Box (nm)"]).split(" ")[0])).toBeCloseTo(
      2 * Number(report["Solute radius (nm)"]) + 0.4,
      3,
    );

    // --- The deviations are on the page ----------------------------------
    // AGENTS.md: the physics never changes silently, and a deviation recorded
    // only in the repository is silent to whoever is running the simulation.
    const shown = await page.locator("#deviations").textContent();
    expect(shown).toContain("deviation 16");
    expect(shown).toContain("deviation 18");
    expect(shown).toContain("deviation 19");
    // 1L2Y has no histidine, so 17 must *not* be listed: a list that includes
    // what did not happen teaches people to skip it.
    expect(shown).not.toContain("deviation 17");

    // --- The artefacts ----------------------------------------------------
    const artefacts = result?.artefacts ?? [];
    expect(artefacts.map((a) => a.name).sort()).toEqual([
      "state.xml",
      "system.pdb",
      "system.xml",
    ]);
    for (const artefact of artefacts) expect(artefact.bytes).toBeGreaterThan(1000);

    expect(failures, `console errors: ${failures.join(" | ")}`).toEqual([]);
    console.log(
      `\n  prepared ${report["Total atoms"]} atoms in a ${report["Box (nm)"]} nm box, ` +
        `${artefacts.reduce((n, a) => n + a.bytes, 0) / 1024 | 0} kB of artefacts`,
    );
  });

  test("prepares a protein-ligand system from uploaded parameters", async ({ page }) => {
    // Phase 4's user-facing path. The parameters themselves are gated against
    // OpenMM in test/prep-parity.test.ts on the same ligand; this checks the
    // page reaches it and handles the two traps that only appear here.
    const ligandXml = "fixtures/ligand/JZ4-ligand.xml";
    test.skip(!existsSync(ligandXml), `${ligandXml} not generated`);

    await page.goto("/prepare.html");
    await page.waitForFunction(() => window.__mirPrepare?.state === "idle");
    await page.locator("#file-pdb").setInputFiles("fixtures/3HTB/system.pdb");
    // The file is read asynchronously, so wait for the summary before asserting
    // on anything the read populates.
    await expect(page.locator("#structure-summary")).toContainText("atoms");

    // 🔴 The heterogen picker, which exists because inference gets this wrong.
    // The notebook renames every ligand to `UNK`, and OpenMM's hydrogens.xml
    // *has* a `UNK` entry -- an unknown amino acid. A ligand called UNK is
    // therefore judged to have rebuildable hydrogens, loses all twelve of
    // them, and matches no template.
    expect(await page.locator("#ligand-residue").inputValue()).toBe("UNK");
    await expect(page.locator("#structure-summary")).toContainText("Dropped:");

    // Selecting a ligand without parameters must fail loudly, not silently.
    await page.locator("#prepare").click();
    await page.waitForFunction(
      () => ["done", "failed"].includes(window.__mirPrepare?.state ?? ""),
      null,
      { timeout: 120_000 },
    );
    expect(await page.evaluate(() => window.__mirPrepare?.state)).toBe("failed");
    expect(await page.evaluate(() => window.__mirPrepare?.error)).toContain(
      "no parameters are loaded",
    );

    await page.locator("#file-ligand").setInputFiles(ligandXml);
    await expect(page.locator("#ligand-summary")).toContainText("residue template");
    // The template is named after the molecule's SMILES, which is why matching
    // is by graph -- the page says so rather than leaving it to look wrong.
    await expect(page.locator("#ligand-summary")).toContainText(
      "matched by the shape of its bonds",
    );
    await expect(page.locator("#structure-summary")).toContainText("Kept:");

    await page.locator("#padding").fill("10");
    await page.locator("#prepare").click();
    await page.waitForFunction(
      () => ["done", "failed"].includes(window.__mirPrepare?.state ?? ""),
      null,
      { timeout: 300_000 },
    );
    const result = await page.evaluate(() => window.__mirPrepare);
    expect(result?.error ?? "").toBe("");
    const report = result?.report ?? {};

    // 🔴 The ligand keeps all 22 atoms. Ten would mean its hydrogens were
    // stripped, which is the failure the picker exists to prevent.
    expect(report["Ligand UNK atoms"]).toBe(22);
    // Protein hydrogens are still rebuilt -- 1313 of them on T4 lysozyme.
    expect(Number(report["Hydrogens added"])).toBeGreaterThan(1000);
    expect(Number(report["Solute charge (e)"])).toBe(6);
    expect(Number(report["Total charge (e)"])).toBeCloseTo(0, 6);
    expect(Number(report["Total atoms"])).toBeGreaterThan(30000);

    const artefacts = (result?.artefacts ?? []).map((a) => a.name).sort();
    expect(artefacts).toEqual(["state.xml", "system.pdb", "system.xml"]);
    console.log(
      `\n  prepared ${report["Total atoms"]} atoms with ligand UNK ` +
        `(${report["Ligand UNK atoms"]} atoms), box ${report["Box (nm)"]} nm`,
    );
  });

  test("the system.xml it offers is a system the engine can read", async ({ page }) => {
    // 🔴 Round-tripped *in the page*, through the same serialiser and parser
    // the engine uses. A prepared system the user cannot take away is a result
    // they have to trust, so the artefact being readable is the deliverable,
    // not a detail.
    await page.goto("/prepare.html");
    await page.waitForFunction(() => window.__mirPrepare?.state === "idle");
    await page.locator("#file-pdb").setInputFiles(receptor);
    // A smaller box, because this test is about the artefact and not the water.
    await page.locator("#padding").fill("1");
    await page.locator("#concentration").fill("0");
    await page.locator("#prepare").click();
    await page.waitForFunction(
      () =>
        window.__mirPrepare?.state === "done" ||
        window.__mirPrepare?.state === "failed",
      null,
      { timeout: 180_000 },
    );
    expect(await page.evaluate(() => window.__mirPrepare?.state)).toBe("done");

    const download = page.waitForEvent("download");
    await page.locator('[data-download="system.xml"]').click();
    const saved = await download;
    const path = await saved.path();
    expect(path).not.toBeNull();
    const xml = readFileSync(path!, "utf8");
    expect(xml).toContain('<System openmmVersion=');
    expect(xml).toContain('type="NonbondedForce"');

    // Parsed with the engine's own reader -- the same function the run page
    // calls on an uploaded file -- and checked against what the page reported.
    const parsed = parseSystem(xml);
    const report = await page.evaluate(() => window.__mirPrepare?.report ?? {});
    expect(parsed.particleCount).toBe(report["Total atoms"]);
    expect(parsed.constraintCount).toBe(report["Constraints"]);
    const types = parsed.forces.map((force) => force.type);
    expect(types).toContain("NonbondedForce");
    expect(types).toContain("PeriodicTorsionForce");
    expect(types).toContain("HarmonicBondForce");
    expect(types).toContain("HarmonicAngleForce");
  });
});
