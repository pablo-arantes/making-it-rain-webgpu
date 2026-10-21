/**
 * The ligand analyses, against MDAnalysis — cells 33 to 37 of the notebook.
 *
 * `tools/dump_ligand_analyses.py` computes each with the tool the notebook
 * uses, on a 60-frame 3HTB trajectory with its JZ4 ligand present as `UNK`.
 *
 * Two findings this comparison produced, both recorded in the implementation:
 *
 * 🔴 **The ligand sits in its own pocket unless "protein" is defined
 * positively.** 3HTB's ligand is residue `UNK` numbered **1**, exactly like its
 * `MET` 1, its first water and its first sodium — a PDB residue number is
 * unique only within a chain and insertion code, and this file has four
 * residues numbered 1. A filter that excluded water and ions by name therefore
 * left the ligand in, at zero distance from itself, and the pocket came out
 * with sixteen residues against MDAnalysis's fifteen. Matching MDAnalysis's
 * `protein` selection as a positive list removes the class.
 *
 * 🔴 **The notebook's ligand RMSD is not displacement.** MDAnalysis superposes
 * the selection before measuring, so it reports the ligand's internal
 * conformational change: here it stays under 0.032 nm across 60 frames while
 * the centre of mass moves much further. Deviation 22.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readDcd } from "../src/analysis/dcd.js";
import { readPdb } from "../src/analysis/pdb.js";
import {
  centreOfMassSeries,
  extractFrame,
  ligandHeavyAtoms,
  ligandRmsd,
  massOf,
  pocketResidues,
  residueDistances,
} from "../src/analysis/ligand.js";
import { writePdb } from "../src/analysis/pdb.js";

interface Reference {
  readonly frames: number;
  readonly ligandName: string;
  readonly ligandHeavyAtoms: readonly number[];
  readonly ligandMasses: Readonly<Record<string, number>>;
  readonly ligandCentreOfMass: readonly (readonly number[])[];
  readonly ligandRmsd: readonly number[];
  readonly pocketCutoff_nm: number;
  readonly pocketResidues: readonly number[];
  readonly pocketDistances: Readonly<Record<string, readonly number[]>>;
  readonly namedResidues: readonly number[];
  readonly namedDistances: Readonly<Record<string, readonly number[]>>;
}

const dcd = "fixtures/analysis-3HTB/trajectory.dcd";
const pdb = "fixtures/3HTB/system.pdb";
const refPath = "fixtures/analysis-3HTB/ligand.json";
const present = existsSync(dcd) && existsSync(pdb) && existsSync(refPath);

const worst = (ours: ArrayLike<number>, theirs: readonly number[]): number => {
  expect(ours.length).toBe(theirs.length);
  let value = 0;
  for (let i = 0; i < theirs.length; i++) {
    value = Math.max(value, Math.abs((ours[i] as number) - (theirs[i] as number)));
  }
  return value;
};

describe.skipIf(!present)("the ligand analyses", () => {
  const reference = present
    ? (JSON.parse(readFileSync(refPath, "utf8")) as Reference)
    : undefined;
  const trajectory = present ? readDcd(new Uint8Array(readFileSync(dcd))) : undefined;
  const topology = present ? readPdb(readFileSync(pdb, "utf8")) : undefined;
  const ligand = present
    ? ligandHeavyAtoms(topology!, reference!.ligandName)
    : undefined;

  it("selects the same ligand atoms", () => {
    // `resname UNK and not (name H*)`: 22 atoms with hydrogens, 10 without.
    expect([...ligand!]).toEqual([...reference!.ligandHeavyAtoms]);
    expect(ligand!).toHaveLength(10);
  });

  it("uses the masses MDAnalysis guessed", () => {
    // Centre of mass is mass-weighted, so a different mass table lands
    // somewhere else. The reference records what it used rather than leaving
    // this to be assumed.
    for (const [atom, mass] of Object.entries(reference!.ligandMasses)) {
      const element = (topology!.atoms[Number(atom)] as { element: string }).element;
      expect(massOf(element)).toBeCloseTo(mass, 6);
    }
  });

  it("computes the ligand RMSD MDAnalysis computes", () => {
    expect(worst(ligandRmsd(trajectory!, ligand!), reference!.ligandRmsd)).toBeLessThan(
      5e-7,
    );
  });

  it("measures conformation, not displacement — and the two differ here", () => {
    // 🔴 The finding worth asserting, not just documenting. If the RMSD were
    // displacement it would track the centre of mass; it does not, because
    // MDAnalysis superposes the selection first.
    const rmsd = ligandRmsd(trajectory!, ligand!);
    const centre = centreOfMassSeries(trajectory!, topology!, ligand!);
    let furthest = 0;
    for (let frame = 0; frame < trajectory!.frameCount; frame++) {
      furthest = Math.max(
        furthest,
        Math.hypot(
          (centre[frame * 3] as number) - (centre[0] as number),
          (centre[frame * 3 + 1] as number) - (centre[1] as number),
          (centre[frame * 3 + 2] as number) - (centre[2] as number),
        ),
      );
    }
    const peakRmsd = Math.max(...rmsd);
    expect(peakRmsd).toBeLessThan(0.05);
    // The centre of mass moves further than the RMSD suggests anything moved.
    expect(furthest).toBeGreaterThan(peakRmsd);
  });

  it("computes the same centre of mass", () => {
    const centre = centreOfMassSeries(trajectory!, topology!, ligand!);
    let value = 0;
    for (let frame = 0; frame < trajectory!.frameCount; frame++) {
      for (let axis = 0; axis < 3; axis++) {
        value = Math.max(
          value,
          Math.abs(
            (centre[frame * 3 + axis] as number) -
              (reference!.ligandCentreOfMass[frame]![axis] as number),
          ),
        );
      }
    }
    expect(value).toBeLessThan(5e-7);
  });

  it("finds the same pocket, from the first frame", () => {
    // 🔴 Sixteen against fifteen before `protein` was defined positively: the
    // ligand is residue UNK *1*, and so are a MET, a water and a sodium.
    const pocket = pocketResidues(
      topology!,
      ligand!,
      trajectory!.positions,
      trajectory!.particleCount,
      0,
      reference!.pocketCutoff_nm,
    );
    expect([...pocket]).toEqual([...reference!.pocketResidues]);
    expect(pocket).toHaveLength(15);
    // The ligand's own residue number must not be in its pocket.
    expect(pocket).not.toContain(1);
  });

  it("computes the same pocket and named-residue distances", () => {
    for (const series of residueDistances(
      trajectory!,
      topology!,
      ligand!,
      reference!.pocketResidues,
    )) {
      const want = reference!.pocketDistances[String(series.residue)];
      expect(want, `no reference for residue ${series.residue}`).toBeDefined();
      expect(worst(series.distances, want!)).toBeLessThan(5e-7);
    }

    // Cell 37, at its own default of residues 78 and 84 — both of which are in
    // the pocket, so the two cells agree where they overlap.
    const named = residueDistances(
      trajectory!,
      topology!,
      ligand!,
      reference!.namedResidues,
    );
    expect(named).toHaveLength(reference!.namedResidues.length);
    for (const series of named) {
      expect(
        worst(series.distances, reference!.namedDistances[String(series.residue)]!),
      ).toBeLessThan(5e-7);
      // The mean and spread the notebook prints, derived from the same series.
      let mean = 0;
      for (const value of series.distances) mean += value;
      expect(series.mean).toBeCloseTo(mean / series.distances.length, 12);
      expect(series.standardDeviation).toBeGreaterThan(0);
    }
  });

  it("extracts one frame without the solvent", () => {
    // Cell 33. The ligand must survive -- it is the point of the file -- while
    // water and ions go.
    const frame = extractFrame(topology!, trajectory!, 42);
    expect(frame.atoms.length).toBeLessThan(topology!.atoms.length / 5);
    expect(frame.residues.some((r) => r.name === "HOH")).toBe(false);
    expect(frame.residues.some((r) => r.name === reference!.ligandName)).toBe(true);
    // Coordinates are frame 42's, not the topology's.
    const base = 42 * trajectory!.particleCount * 3;
    expect(frame.positions[0]).toBeCloseTo(trajectory!.positions[base]!, 6);
    // And it writes as a PDB that reads back with the same atom count.
    const text = writePdb(frame, { remark: "FRAME 42" });
    expect(readPdb(text).atoms).toHaveLength(frame.atoms.length);
  });
});
