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

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
  countAtoms,
  PdbAtomCountMismatch,
  rewritePdb,
} from "../src/analysis/pdb.js";
import type { Vec3 } from "../src/engine/system.js";
import { fixture } from "./fixtures.js";

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
      box: [2.5, 3.5, 4.5] as unknown as Vec3,
    });
    const cryst = rewritten.split("\n").find((l) => l.startsWith("CRYST1")) as string;
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
        box: [3.1, 3.2, 3.3] as unknown as Vec3,
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
