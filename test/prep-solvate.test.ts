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
  cellListFits,
  KNOWN_BOX_SHAPES,
  SHELL_CELLS_PER_AXIS,
  solvate,
  squaredPeriodicDistance,
  UnsupportedBoxShape,
  VDW_RADIUS_PER_SIGMA,
} from "../src/prep/solvate.js";
import {
  type BoxVectors,
  boxVectorsForShape,
  boxVolume,
  isReducedForm,
  maximumCutoff,
  minimumImage,
  perpendicularWidths,
} from "../src/reference/triclinic.js";

interface Dump {
  readonly padding_nm: number;
  readonly boxShape?: string;
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

/**
 * The second water model the plan names.
 *
 * Worth a separate pass rather than a parameter on the first: the model
 * changes the clash radius (SPC/E's sigma is 0.31657 nm against TIP3P's
 * 0.31508) *and* the pre-equilibrated box that gets tiled, so a hard-coded
 * radius or the wrong box file would still pass every TIP3P test. It fits 716
 * waters where TIP3P fits 699 in the same cell.
 */
describe.skipIf(
  !existsSync("fixtures/prep/1L2Y-spce-solvent.json") ||
    !existsSync("vendor/water/spce.pdb"),
)("solvating 1L2Y with SPC/E", () => {
  const dump = JSON.parse(
    readFileSync("fixtures/prep/1L2Y-spce-solvent.json", "utf8"),
  ) as Dump;
  const field = mergeForceFields([
    parseForceField(readFileSync("vendor/ffxml/protein.ff14SB.xml", "utf8")),
    parseForceField(readFileSync("vendor/ffxml/amber14-spce.xml", "utf8")),
  ]);
  const topology = readPdb(readFileSync(pdbPath, "utf8"));
  const bonds = inferBonds(
    topology,
    parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
  );
  const atoms = assignAtoms(topology, bonds, field).atoms;
  const waterBox = readPdb(readFileSync("vendor/water/spce.pdb", "utf8"));

  it("keeps exactly the waters OpenMM kept, in the same places", () => {
    const result = solvate(topology, bonds, atoms, waterBox, {
      padding: dump.padding_nm,
      model: "spce",
      neutralize: false,
      ionicStrength: 0,
    });
    expect(result.waters).toBe(dump.noIons.waters);
    expect(result.topology.atoms).toHaveLength(dump.noIons.atoms);

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
    let worst = 0;
    for (const [index, oxygen] of ours.entries()) {
      for (let axis = 0; axis < 3; axis++) {
        worst = Math.max(worst, Math.abs(oxygen[axis]! - dump.noIons.oxygens[index]![axis]!));
      }
    }
    expect(worst).toBeLessThan(1e-12);
  });

  it("fits more water than TIP3P in the same box, as its radius implies", () => {
    // A sanity check on the radius actually being used: SPC/E's sigma is
    // larger, so each water excludes more volume near the solute -- and yet
    // the pre-equilibrated SPC/E box is denser, which is what wins. 716
    // against 699. If the model were ignored both would be 699.
    const tip3p = JSON.parse(readFileSync(dumpPath, "utf8")) as Dump;
    expect(dump.noIons.waters).toBeGreaterThan(tip3p.noIons.waters);
  });

  it("adds the same ions, which depend only on charge and water count", () => {
    const result = solvate(topology, bonds, atoms, waterBox, {
      padding: dump.padding_nm,
      model: "spce",
      ionicStrength: dump.concentration_molar,
      neutralize: true,
      seed: 24301,
    });
    expect(result.positiveIons).toBe(dump.withIons.ionCounts["NA"]);
    expect(result.negativeIons).toBe(dump.withIons.ionCounts["CL"]);
    expect(result.waters).toBe(dump.withIons.waters);
  });
});

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
    expect(system.boxVectors[0]![0]).toBeCloseTo(solvated.boxVectors[0]![0]!, 12);
    // Every water's two O-H bonds become constraints under h-bonds, so the
    // force keeps only what is left.
    expect(system.constraintCount).toBeGreaterThan(2 * solvated.waters);
  });

  it("uses OpenMM's radius constant, and refuses only what OpenMM lacks", () => {
    expect(VDW_RADIUS_PER_SIGMA).toBeCloseTo(0.5612310241546864907, 15);
    expect(KNOWN_BOX_SHAPES).toEqual(["cube", "dodecahedron", "octahedron"]);
    for (const shape of KNOWN_BOX_SHAPES) {
      expect(() =>
        solvate(topology!, bonds!, atoms!, waterBox!, { padding: 0.4, boxShape: shape }),
      ).not.toThrow();
    }
    // 🔴 Still a refusal rather than a fallback. Silently producing a cube for
    // a shape OpenMM does not have would change the volume by 30%, and the
    // density with it, which nothing downstream would report.
    expect(() =>
      solvate(topology!, bonds!, atoms!, waterBox!, {
        padding: 0.4,
        boxShape: "hexagonal-prism",
      }),
    ).toThrow(UnsupportedBoxShape);
  });

  it("applies the shape to a padding and never to a box given outright", () => {
    // 🔴 `addSolvent` consults `boxShape` in the padding branch alone. A caller
    // who asks for a 3 nm box and a dodecahedron gets the 3 nm box, because
    // reshaping a size they stated would cut the volume they asked for by 30%.
    const explicit = solvate(topology!, bonds!, atoms!, waterBox!, {
      boxSize: [3, 3, 3],
      boxShape: "dodecahedron",
    });
    expect(explicit.boxVectors).toEqual([
      [3, 0, 0],
      [0, 3, 0],
      [0, 0, 3],
    ]);
  });

  it("reduces clash distances exactly as the reference does", () => {
    // Pins the copy in `squaredPeriodicDistance` to `minimumImage`, which is
    // the tested one. The copy exists only to drop the returned array from a
    // loop that runs millions of times, so any difference is a bug.
    for (const shape of KNOWN_BOX_SHAPES) {
      const box: BoxVectors = boxVectorsForShape(shape, 3.1763);
      let worst = 0;
      let seed = 1;
      const next = () => {
        // A plain LCG, so the comparison is the same on every run.
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        return (seed / 0x7fffffff) * 9 - 3;
      };
      for (let trial = 0; trial < 4000; trial++) {
        const a: [number, number, number] = [next(), next(), next()];
        const b: [number, number, number] = [next(), next(), next()];
        const [dx, dy, dz] = minimumImage(b, a, box);
        const expected = dx! * dx! + dy! * dy! + dz! * dz!;
        worst = Math.max(worst, Math.abs(squaredPeriodicDistance(a, b, box) - expected));
      }
      expect(worst).toBe(0);
    }
  });
});

const dodecaDumpPath = "fixtures/prep/1L2Y-solvent-dodecahedron.json";
const dodecaPresent = present && existsSync(dodecaDumpPath);

/**
 * The same comparison as the cube, in a cell that is not rectangular.
 *
 * 🔴 Worth its own block because the cube cannot test any of what makes a
 * skewed cell different. `addSolvent` tiles water over the rectangle the box
 * vectors' **diagonal** spans -- which for a lower-triangular cell is a
 * fundamental domain of the lattice, same volume, one image per point -- and
 * the only places the skew is visible at all are the clash distance and the
 * neighbour grid's fold. Both are wrong in ways a cube never exercises: the
 * grid used to step cell indices to find the 27 neighbours, which wraps z
 * without the x and y shift that crossing the `c` face carries.
 */
describe.skipIf(!dodecaPresent)("solvating 1L2Y in a dodecahedron", () => {
  const dump = dodecaPresent
    ? (JSON.parse(readFileSync(dodecaDumpPath, "utf8")) as Dump)
    : undefined;
  const field = dodecaPresent
    ? mergeForceFields([
        parseForceField(readFileSync("vendor/ffxml/protein.ff14SB.xml", "utf8")),
        parseForceField(readFileSync("vendor/ffxml/amber14-tip3p.xml", "utf8")),
      ])
    : undefined;
  const topology = dodecaPresent ? readPdb(readFileSync(pdbPath, "utf8")) : undefined;
  const bonds = dodecaPresent
    ? inferBonds(
        topology!,
        parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
      )
    : undefined;
  const atoms = dodecaPresent ? assignAtoms(topology!, bonds!, field!).atoms : undefined;
  const waterBox = dodecaPresent ? readPdb(readFileSync(waterPath, "utf8")) : undefined;

  const run = (ions: boolean) =>
    solvate(topology!, bonds!, atoms!, waterBox!, {
      padding: dump!.padding_nm,
      boxShape: "dodecahedron",
      ionicStrength: ions ? dump!.concentration_molar : 0,
      neutralize: ions,
      seed: 24301,
    });

  it("builds the vectors OpenMM builds, in reduced form", () => {
    expect(dump!.boxShape).toBe("dodecahedron");
    const result = run(false);
    for (let row = 0; row < 3; row++) {
      for (let axis = 0; axis < 3; axis++) {
        expect(result.boxVectors[row]![axis]!).toBeCloseTo(
          dump!.noIons.boxVectors[row]![axis]!,
          12,
        );
      }
    }
    // The engine's kernels require this, and both shapes sit exactly on the
    // |cx| <= ax/2 boundary rather than inside it.
    expect(isReducedForm(result.boxVectors)).toBe(true);
    // 70.711% of the enclosing cube, which is the whole reason for the shape.
    const width = result.boxVectors[0]![0]!;
    expect(boxVolume(result.boxVectors) / width ** 3).toBeCloseTo(Math.SQRT1_2, 12);
  });

  it("keeps exactly the waters OpenMM kept, in the same places", () => {
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
    expect(worst).toBeLessThan(1e-12);
  });

  it("adds the ions OpenMM added", () => {
    const result = run(true);
    expect(result.positiveIons).toBe(dump!.withIons.ionCounts["NA"]);
    expect(result.negativeIons).toBe(dump!.withIons.ionCounts["CL"]);
    expect(result.waters).toBe(dump!.withIons.waters);
    expect(result.topology.atoms).toHaveLength(dump!.withIons.atoms);
  });

  it("holds less water than the cube, in proportion to the volume", () => {
    // Not a parity check -- a statement of what the shape buys. The water
    // *density* has to survive the change of shape, because a wrong minimum
    // image in the clash tests shows up there and nowhere else.
    const cube = JSON.parse(readFileSync(dumpPath, "utf8")) as Dump;
    const volumeOf = (v: readonly (readonly number[])[]) =>
      v[0]![0]! * v[1]![1]! * v[2]![2]!;
    const dodecaDensity = dump!.noIons.waters / volumeOf(dump!.noIons.boxVectors);
    const cubeDensity = cube.noIons.waters / volumeOf(cube.noIons.boxVectors);
    expect(dodecaDensity / cubeDensity).toBeGreaterThan(0.95);
    expect(dodecaDensity / cubeDensity).toBeLessThan(1.05);
  });

  it("carries a cutoff for its narrowest perpendicular width, not its edge", () => {
    // 🔴 The padding trap, as a test. This cell's cutoff bound is comfortable,
    // but its narrowest *perpendicular* width is 0.707 of its edge -- so the
    // engine's 27-cell shell, which needs three cells across each width, needs
    // a far wider box than the cube does. At the notebook's 0.4 nm padding a
    // dodecahedron is refused by the engine while the cube of the same padding
    // runs; `docs/DEVIATIONS.md` deviation 19 has the arithmetic.
    const result = run(false);
    const widths = perpendicularWidths(result.boxVectors);
    const edge = result.boxVectors[0]![0]!;
    expect(Math.min(...widths) / edge).toBeCloseTo(Math.SQRT1_2, 9);
    expect(maximumCutoff(result.boxVectors)).toBeCloseTo(0.5 * Math.min(...widths), 12);
    // Three cells across the narrowest width at a 1 nm cutoff needs 4.243 nm.
    expect(edge).toBeLessThan(3 * Math.SQRT2 * 1.0);
  });
});

/**
 * The box floor the engine imposes, asked at preparation time.
 *
 * Needs no fixture, so it runs in every lane. The point of the first assertion
 * is that `src/prep/` states this constant itself rather than importing it from
 * `src/gpu/`, which would put the kernels in the preparation page's bundle for
 * one integer -- so something has to hold the two equal, and this is it.
 */
describe("the cell list's box floor", () => {
  it("mirrors the engine's own minimum, by value", async () => {
    const { MINIMUM_CELLS_PER_AXIS } = await import("../src/gpu/nonbonded.js");
    expect(SHELL_CELLS_PER_AXIS).toBe(MINIMUM_CELLS_PER_AXIS);
  });

  it("measures the perpendicular widths, so a cube and a dodecahedron differ", () => {
    // 🔴 Deviation 19, as arithmetic. The same 3.08 nm width -- 1L2Y at the
    // notebook's 0.4 nm padding -- is a legal cube and an illegal
    // dodecahedron, because the dodecahedron's narrowest perpendicular width
    // is 0.707 of its edge and the shell has to reach across that.
    const width = 3.0806;
    expect(cellListFits(boxVectorsForShape("cube", width), 1).fits).toBe(true);
    expect(cellListFits(boxVectorsForShape("dodecahedron", width), 1).fits).toBe(false);

    // And what it would take: 3*sqrt(2)*cutoff for a dodecahedron, 3 for a cube.
    expect(cellListFits(boxVectorsForShape("dodecahedron", width), 1).width).toBeCloseTo(
      3 * Math.SQRT2,
      9,
    );
    expect(cellListFits(boxVectorsForShape("cube", width), 1).width).toBeCloseTo(3, 9);
    // A box at exactly the returned width fits, which is what the page tells
    // the user to aim for.
    for (const shape of KNOWN_BOX_SHAPES) {
      const needed = cellListFits(boxVectorsForShape(shape, width), 1).width;
      expect(cellListFits(boxVectorsForShape(shape, needed), 1).fits).toBe(true);
    }
  });

  it("scales with the cutoff, not with the shape alone", () => {
    const octahedron = boxVectorsForShape("octahedron", 4.0);
    expect(cellListFits(octahedron, 1).fits).toBe(true);
    expect(cellListFits(octahedron, 1.2).fits).toBe(false);
  });
});
