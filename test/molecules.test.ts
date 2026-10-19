/**
 * The molecule decomposition against OpenMM's `getMolecules()`.
 *
 * The barostat scales each molecule's centroid and translates its atoms
 * rigidly, so this grouping decides the move. Getting it wrong is not a
 * rounding error: merging two molecules moves them as one rigid body, and
 * splitting one stretches it.
 *
 * Labels are canonicalised by lowest atom index on both sides, so two correct
 * decompositions compare equal regardless of the order their union-find ran in.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { findMolecules } from "../src/engine/molecules.js";
import { readNpy } from "./npy.js";
import { fixture, FIXTURE_NAMES } from "./fixtures.js";

interface MoleculeFacts {
  readonly particles: number;
  readonly molecules: number;
  readonly largest_molecule: number;
  readonly size_histogram: Record<string, number>;
}

for (const name of FIXTURE_NAMES) {
  const f = fixture(name);
  const factsPath = `fixtures/${name}/molecule-facts.json`;
  const captured = existsSync(factsPath);

  describe.skipIf(f === null || !captured)(`molecules for ${name}`, () => {
    const facts = captured
      ? (JSON.parse(readFileSync(factsPath, "utf8")) as MoleculeFacts)
      : undefined;

    it("partitions the atoms the way OpenMM does", () => {
      const ours = findMolecules(f!.system);
      const theirs = readNpy(`fixtures/${name}/molecule-labels.npy`).data;
      expect(ours.count).toBe(facts!.molecules);
      expect(ours.largest).toBe(facts!.largest_molecule);
      // The labels themselves, not just the counts: the same number of
      // molecules can still be the wrong partition.
      let differing = 0;
      for (let atom = 0; atom < f!.system.particleCount; atom++) {
        if ((ours.moleculeOf[atom] as number) !== (theirs[atom] as number)) differing++;
      }
      expect(differing).toBe(0);
    });

    it("finds the molecules the system is made of", () => {
      const ours = findMolecules(f!.system);
      const histogram = new Map<number, number>();
      for (let molecule = 0; molecule < ours.count; molecule++) {
        const size =
          (ours.starts[molecule + 1] as number) - (ours.starts[molecule] as number);
        histogram.set(size, (histogram.get(size) ?? 0) + 1);
      }
      for (const [size, expected] of Object.entries(facts!.size_histogram)) {
        expect(histogram.get(Number(size)), `${size}-atom molecules`).toBe(expected);
      }
    });

    it("keeps every constrained pair inside one molecule", () => {
      // Rigid water has constraints and no bonds, so a decomposition built from
      // bonded terms alone would split each one into three molecules and the
      // barostat would pull them apart. This is the check that would catch it.
      const ours = findMolecules(f!.system);
      for (let c = 0; c < f!.system.constraintCount; c++) {
        const a = f!.system.constraintP1[c] as number;
        const b = f!.system.constraintP2[c] as number;
        expect(ours.moleculeOf[a]).toBe(ours.moleculeOf[b]);
      }
    });

    it("lists each molecule's atoms in ascending order, exactly once", () => {
      const ours = findMolecules(f!.system);
      const seen = new Uint8Array(f!.system.particleCount);
      for (let molecule = 0; molecule < ours.count; molecule++) {
        let previous = -1;
        for (
          let slot = ours.starts[molecule] as number;
          slot < (ours.starts[molecule + 1] as number);
          slot++
        ) {
          const atom = ours.atoms[slot] as number;
          expect(atom).toBeGreaterThan(previous);
          previous = atom;
          expect(seen[atom]).toBe(0);
          seen[atom] = 1;
          expect(ours.moleculeOf[atom]).toBe(molecule);
        }
      }
      expect(seen.every((v) => v === 1)).toBe(true);
    });
  });
}
