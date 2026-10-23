/**
 * Rewriting a PDB, and having OpenMM read it back.
 *
 * Each stage writes a PDB of its final frame as the analysis topology. The
 * engine models no topology of its own -- names, residues, chains and elements
 * come from the uploaded file -- so this substitutes coordinates rather than
 * generating a file, and what must survive is everything it did not touch.
 *
 * Column positions are the trap. A PDB is fixed-width, not whitespace
 * delimited: `CA` and `1HB` are told apart by which column the name starts in,
 * and a four-character name leaves no space before the residue. So the test
 * asserts the untouched columns are byte-identical, not merely parseable.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  countAtoms,
  PdbAtomCountMismatch,
  readPdb,
  rewritePdb,
  writePdb,
} from "../src/analysis/pdb.js";
import type { Vec3 } from "../src/engine/system.js";
import { fixture } from "./fixtures.js";
import {
  angleCosines,
  boxVectorsForShape,
  boxVolume,
  edgeLengths,
  isReducedForm,
} from "../src/reference/triclinic.js";

/** A rectangular cell from three edge lengths. */
const rectangular = (lengths: Vec3): readonly [Vec3, Vec3, Vec3] => [
  [lengths[0] as number, 0, 0],
  [0, lengths[1] as number, 0],
  [0, 0, lengths[2] as number],
];

const f = fixture("1L2Y");
const original = f === null ? null : readFileSync("fixtures/1L2Y/system.pdb", "utf8");

describe.skipIf(f === null)("rewritePdb", () => {
  it("counts the atoms the system has", () => {
    expect(countAtoms(original as string)).toBe(f!.system.particleCount);
  });

  it("keeps every column it does not own", () => {
    const shifted = Float64Array.from(f!.state.positions, (v) => v + 0.5);
    const rewritten = rewritePdb(original as string, shifted);
    const before = (original as string).split("\n");
    const after = rewritten.split("\n");
    expect(after).toHaveLength(before.length);
    let checked = 0;
    for (let i = 0; i < before.length; i++) {
      if (!before[i]!.startsWith("ATOM") && !before[i]!.startsWith("HETATM")) {
        expect(after[i], `line ${i}`).toBe(before[i]);
        continue;
      }
      // Columns 1-30 and 55-80, one-based: serial, name, altLoc, residue,
      // chain, sequence, then occupancy, B-factor, element and charge.
      expect(after[i]!.slice(0, 30)).toBe(before[i]!.slice(0, 30));
      expect(after[i]!.slice(54)).toBe(before[i]!.padEnd(54, " ").slice(54));
      checked++;
    }
    expect(checked).toBe(f!.system.particleCount);
  });

  it("writes ångström to three decimals", () => {
    const positions = Float64Array.from(f!.state.positions);
    positions[0] = 1.23456789;
    const rewritten = rewritePdb(original as string, positions);
    const first = rewritten.split("\n").find((l) => l.startsWith("ATOM")) as string;
    // 1.23456789 nm is 12.3456789 A, which the field holds as 12.346.
    expect(first.slice(30, 38)).toBe("  12.346");
  });

  it("replaces CRYST1 when given a box", () => {
    const rewritten = rewritePdb(original as string, f!.state.positions, {
      box: rectangular([2.5, 3.5, 4.5]),
    });
    const cryst = rewritten.split("\n").find((l: string) => l.startsWith("CRYST1")) as string;
    expect(cryst.slice(6, 15)).toBe("   25.000");
    expect(cryst.slice(15, 24)).toBe("   35.000");
    expect(cryst.slice(24, 33)).toBe("   45.000");
    expect(cryst.slice(33, 40)).toBe("  90.00");
  });

  it("refuses a PDB whose atom count does not match", () => {
    // Substitution is by position, so a mismatched file puts every coordinate
    // on the wrong atom -- silently, and in a way no energy check would catch
    // because the result is a different valid structure.
    expect(() => rewritePdb(original as string, new Float64Array(9))).toThrow(
      PdbAtomCountMismatch,
    );
  });

  it("refuses a coordinate the format cannot hold", () => {
    const positions = Float64Array.from(f!.state.positions);
    positions[0] = 10_000; // 100,000 A
    expect(() => rewritePdb(original as string, positions)).toThrow(/does not fit/);
  });

  it("writes a file for OpenMM to read back", () => {
    const dir = "fixtures/pdb-roundtrip";
    mkdirSync(dir, { recursive: true });
    // Coordinates OpenMM can be asked about, including a negative.
    const positions = Float64Array.from(f!.state.positions, (v, i) =>
      i === 0 ? -1.2345 : v,
    );
    writeFileSync(
      `${dir}/ours.pdb`,
      rewritePdb(original as string, positions, {
        box: rectangular([3.1, 3.2, 3.3]),
        remark: "REWRITTEN BY MAKING-IT-RAIN-WEBGPU",
      }),
    );
    writeFileSync(
      `${dir}/expected.json`,
      JSON.stringify(
        {
          atoms: f!.system.particleCount,
          // What a PDB can hold: 0.001 A, so 1e-4 nm.
          tolerance_nm: 1e-4,
          box: [3.1, 3.2, 3.3],
          positions: Array.from(positions),
        },
        null,
        1,
      ),
    );
    expect(countAtoms(readFileSync(`${dir}/ours.pdb`, "utf8"))).toBe(
      f!.system.particleCount,
    );
  });
});

const triclinicPdb = "fixtures/dcd-dodecahedron/topology.pdb";
const triclinicPresent = existsSync(triclinicPdb);

/**
 * A `CRYST1` that is not rectangular, against one OpenMM wrote.
 *
 * 🔴 This file is the reason the type changed. `CRYST1 40 40 40 60 60 90` is a
 * rhombic dodecahedron of volume 45.25 nm³, and the reader took the three
 * lengths and dropped the angles -- producing a 4 nm **cube**, 64 nm³, 41% too
 * much volume. Not a subtly wrong shape: a different cell, read out of a file
 * OpenMM had written correctly, with nothing downstream able to tell.
 */
describe.skipIf(!triclinicPresent)("a non-rectangular CRYST1", () => {
  it("reads the cell OpenMM wrote, with the volume the angles imply", () => {
    const topology = readPdb(readFileSync(triclinicPdb, "utf8"));
    const box = topology.box!;
    for (const length of edgeLengths(box)) expect(length).toBeCloseTo(4, 9);
    // 🔴 The number that says the old reader was wrong rather than imprecise.
    // It took the three lengths, dropped the angles, and produced a 4 nm cube
    // of 64 nm³ where the file describes 45.25 -- 41% too much volume, out of
    // a file OpenMM had written correctly.
    expect(boxVolume(box)).toBeCloseTo(45.2548, 4);
    expect(boxVolume(box) / 4 ** 3).toBeCloseTo(Math.SQRT1_2, 9);
    // Lower triangular, which is what the kernels require.
    expect(isReducedForm(box)).toBe(true);
    expect(box[0]![1]).toBe(0);
    expect(box[0]![2]).toBe(0);
    expect(box[1]![2]).toBe(0);
  });

  it("agrees with computePeriodicBoxVectors, 120 degrees and all", () => {
    // 🔴 **A `CRYST1` preserves the lattice, not the representation.** OpenMM
    // wrote `60.00 60.00 90.00`; reading it back gives the *120/120/90* form of
    // the same cell, because `computePeriodicBoxVectors` reduces at the end and
    // `c[1] / b[1]` lands on 0.5000000000000001 rather than an exact tie, so
    // the rounding takes a whole `b`. Measured from OpenMM itself:
    // `computePeriodicBoxVectors(4, 4, 4, 60, 60, 90)` returns
    // `c = (-1.9999999999999996, -1.9999999999999996, 2.8284271247461894)` and
    // `computeLengthsAndAngles` turns that into 120, 120, 90.
    //
    // So this is not a defect to fix on either side: the two forms are the same
    // lattice, the volume is identical, and a dodecahedron's `system.pdb` will
    // say 120/120/90. The test exists because the obvious expectation -- that a
    // file round-trips to itself -- is false, and would otherwise be "fixed"
    // into a disagreement with OpenMM.
    const box = readPdb(readFileSync(triclinicPdb, "utf8")).box!;
    expect(box[2]![0]).toBeCloseTo(-1.9999999999999996, 15);
    expect(box[2]![1]).toBeCloseTo(-1.9999999999999996, 15);
    expect(box[2]![2]).toBeCloseTo(2.8284271247461894, 15);
    const cosines = angleCosines(box);
    expect(cosines[0]).toBeCloseTo(-0.5, 12);
    expect(cosines[1]).toBeCloseTo(-0.5, 12);
    expect(cosines[2]).toBeCloseTo(0, 12);
  });

  it("is a fixed point after one round trip", () => {
    // The 120/120/90 form *is* stable, so writing and reading repeatedly does
    // not keep walking the cell around the lattice.
    const once = readPdb(readFileSync(triclinicPdb, "utf8"));
    const cryst = (t: typeof once) =>
      writePdb(t)
        .split("\n")
        .find((l: string) => l.startsWith("CRYST1")) as string;
    const first = cryst(once);
    const twice = readPdb(writePdb(once));
    expect(cryst(twice)).toBe(first);
    expect(readPdb(writePdb(twice)).box).toEqual(twice.box);
    expect(first.slice(33, 54)).toBe(" 120.00 120.00  90.00");
    expect(boxVolume(twice.box!)).toBeCloseTo(45.2548, 4);
  });

  it("keeps a dodecahedron's volume through CRYST1's two decimals", () => {
    const box = boxVectorsForShape("dodecahedron", 4.25);
    const back = readPdb(
      writePdb({
        atoms: [],
        residues: [],
        chains: [],
        box,
        conect: [],
        positions: new Float64Array(0),
      }),
    ).box!;
    // Same lattice, possibly a different reduced form, so the invariants are
    // the volume and the edge lengths rather than the components.
    expect(boxVolume(back)).toBeCloseTo(boxVolume(box), 3);
    const theirs = edgeLengths(box);
    for (const [axis, length] of edgeLengths(back).entries()) {
      expect(length).toBeCloseTo(theirs[axis]!, 3);
    }
    expect(isReducedForm(back)).toBe(true);
  });
});
