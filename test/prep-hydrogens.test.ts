/**
 * Adding hydrogens, against what `Modeller.addHydrogens` produced.
 *
 * Compared on everything that is well defined: the variant chosen per residue,
 * the full atom list in order with names and elements, and every bond. Not
 * compared on positions, and that is not a concession -- OpenMM places each
 * hydrogen at its parent plus a `random.random()` offset and then minimises the
 * set, so it does not reproduce its own hydrogen positions between runs. There
 * is no target there. `docs/DEVIATIONS.md` records it, and the geometry is
 * instead checked for being physically sensible.
 *
 * Both fixtures start from the deposited receptor with water and every hydrogen
 * removed, which is the same starting point `tools/build_prep_fixture.py` gives
 * OpenMM, so the two are rebuilding from the same heavy atoms.
 *
 * The last test is the one that matters most: feed the hydrogens we added
 * through the whole parameter path and compare the resulting `System` against
 * the one OpenMM built from *its* hydrogens. Parameters do not depend on
 * hydrogen positions, so if our hydrogen set is right the two must agree
 * exactly -- which closes the path from a deposited PDB to a runnable system.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ifPresent } from "./fixtures.js";
import { readPdb, type PdbTopology } from "../src/analysis/pdb.js";
import {
  parseSystem,
  type HarmonicBondForce,
  type NonbondedForce,
  type PeriodicTorsionForce,
} from "../src/engine/system.js";
import { inferBonds, parseStandardBonds } from "../src/prep/bonds.js";
import { parseForceField } from "../src/prep/ffxml.js";
import {
  addHydrogens,
  isHydrogenAtom,
  isWater,
  parseHydrogenDefinitions,
  stripAtoms,
} from "../src/prep/hydrogens.js";
import { prepareSystem } from "../src/prep/prepare.js";
import { buildSystem } from "../src/prep/system.js";

interface Dump {
  readonly strippedAtoms: readonly (readonly string[])[];
  readonly variants: readonly (string | null)[];
  readonly atoms: readonly (readonly [string, string | null, number])[];
  readonly bonds: readonly (readonly [number, number])[];
}

const definitionsPath = "vendor/ffxml/hydrogens.xml";
const residuesPath = "vendor/ffxml/residues.xml";
const fieldPath = "vendor/ffxml/protein.ff14SB.xml";

const FIXTURES = [
  { name: "1L2Y", heavy: 154, atoms: 304, note: "no histidine, no cysteine" },
  { name: "3HTB", heavy: 1301, atoms: 2614, note: "histidine and two cysteines" },
] as const;

/** Residue index per atom, which is what the dump records. */
function residueOf(topology: PdbTopology): readonly number[] {
  const of = new Array<number>(topology.atoms.length);
  for (const [index, residue] of topology.residues.entries()) {
    for (const atom of residue.atoms) of[atom] = index;
  }
  return of;
}

for (const fixture of FIXTURES) {
  const pdbPath = `fixtures/${fixture.name}/receptor.pdb`;
  const dumpPath = `fixtures/prep/${fixture.name}-hydrogens.json`;
  const systemPath = `fixtures/prep/${fixture.name}-system.xml`;
  const present =
    existsSync(pdbPath) &&
    existsSync(dumpPath) &&
    existsSync(definitionsPath) &&
    existsSync(fieldPath);

  describe.skipIf(!present)(`adding hydrogens to ${fixture.name} (${fixture.note})`, () => {
    const definitions = present
      ? parseHydrogenDefinitions(readFileSync(definitionsPath, "utf8"))
      : undefined;
    const dump = present
      ? (JSON.parse(readFileSync(dumpPath, "utf8")) as Dump)
      : undefined;
    const standardBonds = present
      ? parseStandardBonds(readFileSync(residuesPath, "utf8"))
      : undefined;

    const stripped = present
      ? stripAtoms(
          stripAtoms(readPdb(readFileSync(pdbPath, "utf8")), isWater).topology,
          isHydrogenAtom,
        ).topology
      : undefined;
    const result =
      present && stripped !== undefined
        ? addHydrogens(stripped, inferBonds(stripped, standardBonds!), definitions!, 7)
        : undefined;

    it("parses the hydrogen definitions", () => {
      expect(definitions!.size).toBeGreaterThan(30);
      const his = definitions!.get("HIS")!;
      expect(his.variants).toEqual(["HID", "HIE", "HIP", "HIN"]);
      // HD1 belongs to HID and HIP only, and the terminal-only hydrogens carry
      // their flag -- both are what the filters act on.
      expect(his.hydrogens.find((h) => h.name === "HD1")!.variants).toEqual([
        "HID",
        "HIP",
      ]);
      expect(his.hydrogens.find((h) => h.name === "H2")!.terminal).toBe("N");
      expect(definitions!.get("CYS")!.hydrogens.find((h) => h.name === "HG")!.maxPh).toBe(
        8.5,
      );
    });

    it("strips to the same heavy atoms OpenMM started from", () => {
      expect(stripped!.atoms).toHaveLength(fixture.heavy);
      expect(stripped!.atoms).toHaveLength(dump!.strippedAtoms.length);
      const ours = stripped!.atoms.map((a) =>
        [a.name, a.residueName, a.residueId, a.chainId].join("|"),
      );
      expect(ours).toEqual(dump!.strippedAtoms.map((r) => r.join("|")));
    });

    it("chooses the same protonation variant for every residue", () => {
      // 🔴 Includes the histidine. OpenMM's documented "better hydrogen bond"
      // rule cannot select HIE -- `ne2IsBonded` is never assigned, so the
      // automatic path always yields HID -- and we reproduce that deliberately.
      // See src/prep/hydrogens.ts. Disagreeing here would change the charges on
      // a histidine and nothing would flag it.
      const ours = result!.variants.map((v) => v ?? null);
      expect(ours).toEqual(dump!.variants.map((v) => v ?? null));
    });

    it("produces the same atoms, with the same names, in the same order", () => {
      const of = residueOf(result!.topology);
      const ours = result!.topology.atoms.map(
        (a, index) => `${a.name}|${a.element}|${of[index]}`,
      );
      expect(ours).toHaveLength(fixture.atoms);
      expect(ours).toEqual(dump!.atoms.map((r) => `${r[0]}|${r[1]}|${r[2]}`));
    });

    it("produces the same bonds", () => {
      const ours = new Set(
        result!.bonds.map(([a, b]) => `${Math.min(a, b)},${Math.max(a, b)}`),
      );
      const theirs = new Set(dump!.bonds.map(([a, b]) => `${a},${b}`));
      expect([...ours].sort()).toEqual([...theirs].sort());
    });

    it("places each hydrogen 0.1 nm from its parent, away from its neighbours", () => {
      // 🔴 Positions are a deviation, so what is checked is that they are sane
      // rather than that they match -- and this test earned its place. The
      // first implementation summed bond vectors the way OpenMM does, which is
      // degenerate: the first hydrogen on a parent lies along its only bond's
      // reverse, so re-summing leaves the direction unchanged and the second
      // lands on top of the first. 92 coincident pairs across 46 parents on
      // 1L2Y. Coincident atoms are an infinite energy, not a cosmetic flaw,
      // and every other test here passed regardless, because names, order and
      // bonds were all still right.
      //
      // So: the right bond length, no two atoms in one place, and nothing
      // placed into an existing bond. Measured minimum angle 85.9 degrees on
      // 1L2Y and 86.6 on 3HTB, against 109.5 for an ideal tetrahedron.
      const { topology, bonds, added } = result!;
      const parentOf = new Map<number, number>();
      const neighbours: number[][] = topology.atoms.map(() => []);
      for (const [a, b] of bonds) {
        neighbours[a]!.push(b);
        neighbours[b]!.push(a);
      }
      for (const hydrogen of added) {
        const parents = neighbours[hydrogen]!;
        expect(parents).toHaveLength(1);
        parentOf.set(hydrogen, parents[0]!);
      }
      const at = (i: number): [number, number, number] => [
        topology.positions[i * 3]!,
        topology.positions[i * 3 + 1]!,
        topology.positions[i * 3 + 2]!,
      ];
      let worstAngle = 180;
      let coincident = 0;
      for (const hydrogen of added) {
        const parent = parentOf.get(hydrogen)!;
        const h = at(hydrogen);
        const p = at(parent);
        const d = [h[0] - p[0], h[1] - p[1], h[2] - p[2]];
        const length = Math.hypot(d[0]!, d[1]!, d[2]!);
        expect(length).toBeCloseTo(0.1, 9);

        // Angle to each heavy neighbour of the parent: a tetrahedral slot is
        // 109.5 degrees, and anything below 60 means it was placed on top of
        // an existing bond.
        for (const other of neighbours[parent]!) {
          if (other === hydrogen) continue;
          const o = at(other);
          const v = [o[0] - p[0], o[1] - p[1], o[2] - p[2]];
          const vLength = Math.hypot(v[0]!, v[1]!, v[2]!);
          if (vLength === 0) continue;
          const cosine =
            (d[0]! * v[0]! + d[1]! * v[1]! + d[2]! * v[2]!) / (length * vLength);
          const angle = (Math.acos(Math.max(-1, Math.min(1, cosine))) * 180) / Math.PI;
          worstAngle = Math.min(worstAngle, angle);
          if (Math.hypot(h[0] - o[0], h[1] - o[1], h[2] - o[2]) < 1e-6) coincident++;
        }
      }
      expect(coincident).toBe(0);
      expect(worstAngle).toBeGreaterThan(60);
    });

    it("yields the System OpenMM built from its own hydrogens", () => {
      // 🔴 The end-to-end check, and the reason the position deviation is
      // affordable: parameters do not depend on where a hydrogen sits, so if
      // our hydrogen *set* matches, every force must match exactly -- compared
      // against the serialised System built from OpenMM's hydrogens, not from
      // ours. This is the whole path from a deposited PDB to a runnable system.
      if (!existsSync(systemPath)) return;
      const prepared = prepareSystem(
        result!.topology,
        parseForceField(readFileSync(fieldPath, "utf8")),
        standardBonds!,
      );
      const ours = buildSystem(prepared);
      const theirs = parseSystem(readFileSync(systemPath, "utf8"));

      expect([...ours.masses]).toEqual([...theirs.masses]);
      const pick = <T>(system: typeof ours, type: string): T =>
        system.forces.find((f) => f.type === type) as unknown as T;
      const bondsOurs = pick<HarmonicBondForce>(ours, "HarmonicBondForce");
      const bondsTheirs = pick<HarmonicBondForce>(theirs, "HarmonicBondForce");
      expect([...bondsOurs.p1]).toEqual([...bondsTheirs.p1]);
      expect([...bondsOurs.length]).toEqual([...bondsTheirs.length]);
      const torsionsOurs = pick<PeriodicTorsionForce>(ours, "PeriodicTorsionForce");
      const torsionsTheirs = pick<PeriodicTorsionForce>(theirs, "PeriodicTorsionForce");
      expect(torsionsOurs.count).toBe(torsionsTheirs.count);
      expect([...torsionsOurs.p1]).toEqual([...torsionsTheirs.p1]);
      expect([...torsionsOurs.k]).toEqual([...torsionsTheirs.k]);
      const nbOurs = pick<NonbondedForce>(ours, "NonbondedForce");
      const nbTheirs = pick<NonbondedForce>(theirs, "NonbondedForce");
      expect([...nbOurs.charge]).toEqual([...nbTheirs.charge]);
      expect([...nbOurs.epsilon]).toEqual([...nbTheirs.epsilon]);
      expect([...nbOurs.exceptionChargeProd]).toEqual([
        ...nbTheirs.exceptionChargeProd,
      ]);
    });
  });
}


/**
 * pH, which the notebook exposes and which changes the chemistry.
 *
 * Worth its own case because every other test here runs at pH 7, where the
 * acidic variants are all deprotonated and the `maxph` attribute never
 * decides anything. At pH 4 it decides a lot: aspartate and glutamate take
 * their neutral ASH and GLH forms, and histidine goes to HIP. On T4 lysozyme
 * that is 21 extra hydrogens and a different net charge, so a pipeline that
 * ignored the field entirely would pass everything above.
 *
 * `docs/DEVIATIONS.md` entry 8 is about this field: the notebook has a pH
 * control that cannot take effect and a hardcoded 7.4 in the call that can.
 */
const hasPh4 =
  existsSync("fixtures/prep/3HTB-ph4-hydrogens.json") &&
  existsSync("fixtures/3HTB/receptor.pdb");

describe.skipIf(!hasPh4)(
  "adding hydrogens at pH 4",
  () => {
    // `skipIf` does not stop this body running -- see `ifPresent`.
    const dump = ifPresent(
      hasPh4,
      () =>
        JSON.parse(
          readFileSync("fixtures/prep/3HTB-ph4-hydrogens.json", "utf8"),
        ) as Dump,
    );
    const definitions = ifPresent(hasPh4, () =>
      parseHydrogenDefinitions(readFileSync(definitionsPath, "utf8")),
    );
    const standardBonds = ifPresent(hasPh4, () =>
      parseStandardBonds(readFileSync(residuesPath, "utf8")),
    );
    const stripped = ifPresent(
      hasPh4,
      () =>
        stripAtoms(
          stripAtoms(
            readPdb(readFileSync("fixtures/3HTB/receptor.pdb", "utf8")),
            isWater,
          ).topology,
          isHydrogenAtom,
        ).topology,
    );
    const result = ifPresent(hasPh4, () =>
      addHydrogens(stripped, inferBonds(stripped, standardBonds), definitions, 4),
    );

    it("protonates the acids and the histidine that pH 7 leaves bare", () => {
      // 2635 atoms against 2614 at pH 7: the `maxph` attribute on ASH's HD2
      // (4.4) and GLH's HE2 (4.4) now admits them, and HIS becomes HIP
      // through the pH <= 6.5 branch rather than the hydrogen-bond one.
      expect(result.topology.atoms).toHaveLength(2635);
      expect(result.added).toHaveLength(1334);
      expect(result.variants.filter((v) => v === "HIP")).toHaveLength(1);
      expect(result.variants.filter((v) => v === "HID")).toHaveLength(0);
    });

    it("matches OpenMM atom for atom at this pH too", () => {
      const of = residueOf(result.topology);
      const ours = result.topology.atoms.map(
        (a, index) => `${a.name}|${a.element}|${of[index]}`,
      );
      expect(ours).toEqual(dump.atoms.map((r) => `${r[0]}|${r[1]}|${r[2]}`));
      expect(result.variants.map((v) => v ?? null)).toEqual(
        dump.variants.map((v) => v ?? null),
      );
    });

    it("gives a different system from pH 7, which is the point of the field", () => {
      const neutral = addHydrogens(
        stripped,
        inferBonds(stripped, standardBonds),
        definitions,
        7,
      );
      expect(neutral.topology.atoms).toHaveLength(2614);
      expect(result.topology.atoms.length - neutral.topology.atoms.length).toBe(21);
    });
  },
);
