/**
 * Filling a box with water and ions, against `Modeller.addSolvent`.
 *
 * Split the way the algorithm splits. Water placement is exactly reproducible
 * — tile a pre-equilibrated box, keep the molecules whose oxygen lands inside
 * the cell, drop the ones clashing with the solute by summed van der Waals
 * radii, drop the ones clashing only across the periodic boundary — so it is
 * compared position by position. Ion *placement* is not: `_addIons` picks the
 * waters to replace with `random.choice`, so OpenMM does not reproduce its own
 * choice between runs. The ion *counts* follow from the solute charge and the
 * water count, and those are exact. Deviation 18.
 *
 * The reference comes from `tools/dump_solvent.py`, which runs `addSolvent`
 * twice for exactly this reason: once with ions off, once with the notebook's
 * settings.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb, type PdbTopology } from "../src/analysis/pdb.js";
import { assignAtoms } from "../src/prep/assign.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";
import { mergeForceFields, parseForceField } from "../src/prep/ffxml.js";
import { prepareSystem } from "../src/prep/prepare.js";
import { buildSystem } from "../src/prep/system.js";
import { totalCharge } from "../src/prep/assign.js";
import { NonbondedMethod } from "../src/engine/system.js";
import {
  boxWidthForPadding,
  solvate,
  UnsupportedBoxShape,
  VDW_RADIUS_PER_SIGMA,
} from "../src/prep/solvate.js";

interface Dump {
  readonly padding_nm: number;
  readonly concentration_molar: number;
  readonly soluteAtoms: number;
  readonly noIons: {
    readonly boxVectors: readonly (readonly number[])[];
    readonly waters: number;
    readonly atoms: number;
    readonly oxygens: readonly (readonly number[])[];
  };
  readonly withIons: {
    readonly waters: number;
    readonly ionCounts: Readonly<Record<string, number>>;
    readonly atoms: number;
  };
}

const pdbPath = "fixtures/prep/1L2Y.pdb";
const dumpPath = "fixtures/prep/1L2Y-solvent.json";
const waterPath = "vendor/water/tip3p.pdb";
const present =
  existsSync(pdbPath) &&
  existsSync(dumpPath) &&
  existsSync(waterPath) &&
  existsSync("vendor/ffxml/amber14-tip3p.xml");

describe.skipIf(!present)("solvating 1L2Y", () => {
  const dump = present ? (JSON.parse(readFileSync(dumpPath, "utf8")) as Dump) : undefined;
  const field = present
    ? mergeForceFields([
        parseForceField(readFileSync("vendor/ffxml/protein.ff14SB.xml", "utf8")),
        parseForceField(readFileSync("vendor/ffxml/amber14-tip3p.xml", "utf8")),
      ])
    : undefined;
  const topology = present ? readPdb(readFileSync(pdbPath, "utf8")) : undefined;
  const bonds = present
    ? inferBonds(
        topology!,
        parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
      )
    : undefined;
  const atoms = present ? assignAtoms(topology!, bonds!, field!).atoms : undefined;
  const waterBox = present ? readPdb(readFileSync(waterPath, "utf8")) : undefined;

  const run = (ions: boolean) =>
    solvate(topology!, bonds!, atoms!, waterBox!, {
      padding: dump!.padding_nm,
      ionicStrength: ions ? dump!.concentration_molar : 0,
      neutralize: ions,
      seed: 24301,
    });

  it("reads the pre-equilibrated water box", () => {
    expect(waterBox!.residues).toHaveLength(895);
    expect(waterBox!.box).toEqual([3, 3, 3]);
  });

  it("picks the same box width as OpenMM", () => {
    // The bounding *sphere* about the bounding box's centre, plus the padding,
    // and at least twice the padding -- not the bounding box.
    const width = boxWidthForPadding(
      topology!.positions,
      topology!.atoms.length,
      dump!.padding_nm,
    );
    for (let axis = 0; axis < 3; axis++) {
      expect(width).toBeCloseTo(dump!.noIons.boxVectors[axis]![axis]!, 12);
    }
  });

  it("keeps exactly the waters OpenMM kept, in the same places", () => {
    // 🔴 The test that found the one real bug here. A non-periodic neighbour
    // search keeps 855 waters instead of 699 -- a 22% density error -- because
    // a water near one face is a neighbour of the solute's image through the
    // opposite face, and with this box (3.18 nm) and solute (1.39 nm radius)
    // the images sit against every face. Nothing downstream would flag it.
    const result = run(false);
    expect(result.waters).toBe(dump!.noIons.waters);
    expect(result.topology.atoms).toHaveLength(dump!.noIons.atoms);

    const ours: number[][] = [];
    for (const residue of result.topology.residues) {
      if (residue.name !== "HOH") continue;
      const oxygen = residue.atoms.find(
        (a) => (result.topology.atoms[a] as { element: string }).element === "O",
      )!;
      ours.push([
        result.topology.positions[oxygen * 3]!,
        result.topology.positions[oxygen * 3 + 1]!,
        result.topology.positions[oxygen * 3 + 2]!,
      ]);
    }
    ours.sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]! || a[2]! - b[2]!);
    expect(ours).toHaveLength(dump!.noIons.oxygens.length);
    let worst = 0;
    for (const [index, oxygen] of ours.entries()) {
      const theirs = dump!.noIons.oxygens[index]!;
      for (let axis = 0; axis < 3; axis++) {
        worst = Math.max(worst, Math.abs(oxygen[axis]! - theirs[axis]!));
      }
    }
    // f64 round-off on a sum of coordinates, not a tolerance on the algorithm.
    expect(worst).toBeLessThan(1e-12);
  });

  it("adds the ions OpenMM added, and neutralises the same charge", () => {
    const result = run(true);
    // 1L2Y is +1, so one Cl- neutralises it, and 0.15 M over 698 waters is two
    // more pairs: (699 - 1) * 0.15 / 55.4 = 1.89, which rounds to 2.
    expect(result.soluteCharge).toBe(1);
    expect(result.positiveIons).toBe(dump!.withIons.ionCounts["NA"]);
    expect(result.negativeIons).toBe(dump!.withIons.ionCounts["CL"]);
    expect(result.waters).toBe(dump!.withIons.waters);
    expect(result.topology.atoms).toHaveLength(dump!.withIons.atoms);
  });

  it("places ions reproducibly from the seed, and spread apart", () => {
    // Deviation 18: OpenMM uses `random.choice` and we use Threefry on the run
    // seed, so our placement is reproducible and theirs is not. What must hold
    // is that the same seed gives the same answer, a different seed does not,
    // and OpenMM's 0.05 nm separation rule is respected.
    const first = run(true);
    const again = run(true);
    const ionsOf = (result: { topology: PdbTopology }): string =>
      result.topology.residues
        .filter((r) => r.name === "NA" || r.name === "CL")
        .map((r) => {
          const atom = r.atoms[0]!;
          return `${r.name}:${result.topology.positions[atom * 3]!.toFixed(6)}`;
        })
        .join("|");
    expect(ionsOf(again)).toBe(ionsOf(first));

    const other = solvate(topology!, bonds!, atoms!, waterBox!, {
      padding: dump!.padding_nm,
      ionicStrength: dump!.concentration_molar,
      neutralize: true,
      seed: 99,
    });
    expect(ionsOf(other)).not.toBe(ionsOf(first));

    const points = first.topology.residues
      .filter((r) => r.name === "NA" || r.name === "CL")
      .map((r) => {
        const atom = r.atoms[0]!;
        return [
          first.topology.positions[atom * 3]!,
          first.topology.positions[atom * 3 + 1]!,
          first.topology.positions[atom * 3 + 2]!,
        ] as const;
      });
    expect(points).toHaveLength(5);
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        const a = points[i]!;
        const b = points[j]!;
        expect(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])).toBeGreaterThan(0.05);
      }
    }
  });

  it("produces a system the engine can run, from a deposited PDB", () => {
    // 🔴 The whole point of Phase 3, end to end: deposited structure ->
    // hydrogens -> solvation -> parameters -> a `System`. There is no OpenMM
    // reference for a box *we* filled, so what is checked is that every
    // residue we invented matches a template and that the result is neutral --
    // which is the property solvation exists to deliver, and the one a wrong
    // ion count or a mismatched water would break.
    const solvated = run(true);
    const prepared = prepareSystem(
      solvated.topology,
      field!,
      parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
    );
    expect(prepared.assignment.atoms).toHaveLength(solvated.topology.atoms.length);

    // Water and the ions all matched templates; HOH, NA and CL are the ones
    // only this path can reach.
    const templates = new Set(prepared.assignment.matches.map((m) => m.template));
    expect(templates.has("HOH")).toBe(true);
    expect(templates.has("NA")).toBe(true);
    expect(templates.has("CL")).toBe(true);

    // Neutral to within round-off: +1 peptide, 2 Na+, 3 Cl-.
    expect(totalCharge(prepared.assignment)).toBeCloseTo(0, 6);

    const system = buildSystem(prepared, {
      nonbondedMethod: NonbondedMethod.PME,
      cutoff: 1,
      constraints: "h-bonds",
    });
    expect(system.particleCount).toBe(solvated.topology.atoms.length);
    expect(system.boxVectors[0]![0]).toBeCloseTo(solvated.boxVectors[0], 12);
    // Every water's two O-H bonds become constraints under h-bonds, so the
    // force keeps only what is left.
    expect(system.constraintCount).toBeGreaterThan(2 * solvated.waters);
  });

  it("uses OpenMM's radius constant and refuses shapes it cannot do", () => {
    expect(VDW_RADIUS_PER_SIGMA).toBeCloseTo(0.5612310241546864907, 15);
    // A dodecahedron needs a triclinic minimum image that is not written.
    // Quietly producing a cube would change the density.
    expect(() =>
      solvate(topology!, bonds!, atoms!, waterBox!, {
        padding: 0.4,
        boxShape: "dodecahedron",
      }),
    ).toThrow(UnsupportedBoxShape);
  });
});
