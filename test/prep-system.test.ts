/**
 * The built `System`, against the serialised `System` OpenMM built.
 *
 * This is the strongest test of the preparation path, because it compares
 * against the file the engine actually consumes rather than against a summary
 * of it. The engine contract is OpenMM `System` XML -- chosen so the notebook
 * stays the oracle and so a user can upload a system prepared elsewhere -- and
 * preparing in the browser is only useful if it arrives at the *same* object.
 *
 * Everything is compared element by element, in index order, with no
 * tolerance at all: `toEqual` on the typed arrays. That is a deliberately
 * brittle choice and it is affordable because both sides read the same decimal
 * strings out of the same ffxml and do the same arithmetic on them. The one
 * place it bit was the bond order, which is why `inferBonds` now keeps
 * OpenMM's order and direction rather than sorting.
 *
 * Force *order* within the System is not compared. OpenMM's generators run in
 * registration order and ours do not, so its forces come out as bond, torsion,
 * nonbonded, angle. Nothing reads them positionally -- the engine dispatches on
 * `type` -- so matching that would be imitation without a reason.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readPdb } from "../src/analysis/pdb.js";
import {
  parseSystem,
  type HarmonicAngleForce,
  type HarmonicBondForce,
  type NonbondedForce,
  type OpenMMSystem,
  type PeriodicTorsionForce,
} from "../src/engine/system.js";
import { parseStandardBonds } from "../src/prep/bonds.js";
import { parseForceField } from "../src/prep/ffxml.js";
import { prepareSystem } from "../src/prep/prepare.js";
import { buildSystem } from "../src/prep/system.js";

const FIXTURES = ["1L2Y", "3HTB"] as const;
const fieldPath = "vendor/ffxml/protein.ff14SB.xml";

function force<T extends { type: string }>(system: OpenMMSystem, type: string): T {
  const found = system.forces.find((f) => f.type === type);
  if (found === undefined) throw new Error(`no ${type} in the system`);
  return found as unknown as T;
}

for (const name of FIXTURES) {
  const pdbPath = `fixtures/prep/${name}.pdb`;
  const systemPath = `fixtures/prep/${name}-system.xml`;
  const present =
    existsSync(pdbPath) && existsSync(systemPath) && existsSync(fieldPath);

  describe.skipIf(!present)(`the System built for ${name}`, () => {
    const ours = present
      ? buildSystem(
          prepareSystem(
            readPdb(readFileSync(pdbPath, "utf8")),
            parseForceField(readFileSync(fieldPath, "utf8")),
            parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
          ),
        )
      : undefined;
    const theirs = present
      ? parseSystem(readFileSync(systemPath, "utf8"))
      : undefined;

    it("has the same particles, masses and box", () => {
      expect(ours!.particleCount).toBe(theirs!.particleCount);
      expect([...ours!.masses]).toEqual([...theirs!.masses]);
      // 2 nm cube: OpenMM's default for a System with no periodic box, which
      // this fixture has because it is unsolvated.
      expect(ours!.boxVectors).toEqual(theirs!.boxVectors);
    });

    it("has no constraints, as the fixture was built", () => {
      // `createSystem(constraints=None, rigidWater=False)`. Constraints change
      // what is in the System without changing any assignment, so the parity
      // fixture leaves them out; `buildSystem` can add them and that is tested
      // separately below.
      expect(ours!.constraintCount).toBe(0);
      expect(theirs!.constraintCount).toBe(0);
    });

    it("has the same bonds, in the same order and direction", () => {
      const a = force<HarmonicBondForce>(ours!, "HarmonicBondForce");
      const b = force<HarmonicBondForce>(theirs!, "HarmonicBondForce");
      expect(a.count).toBe(b.count);
      expect([...a.p1]).toEqual([...b.p1]);
      expect([...a.p2]).toEqual([...b.p2]);
      expect([...a.length]).toEqual([...b.length]);
      expect([...a.k]).toEqual([...b.k]);
    });

    it("has the same angles, in the same order", () => {
      const a = force<HarmonicAngleForce>(ours!, "HarmonicAngleForce");
      const b = force<HarmonicAngleForce>(theirs!, "HarmonicAngleForce");
      expect(a.count).toBe(b.count);
      expect([...a.p1]).toEqual([...b.p1]);
      expect([...a.p2]).toEqual([...b.p2]);
      expect([...a.p3]).toEqual([...b.p3]);
      expect([...a.angle]).toEqual([...b.angle]);
      expect([...a.k]).toEqual([...b.k]);
    });

    it("has the same torsions, in the same order", () => {
      const a = force<PeriodicTorsionForce>(ours!, "PeriodicTorsionForce");
      const b = force<PeriodicTorsionForce>(theirs!, "PeriodicTorsionForce");
      expect(a.count).toBe(b.count);
      expect([...a.p1]).toEqual([...b.p1]);
      expect([...a.p2]).toEqual([...b.p2]);
      expect([...a.p3]).toEqual([...b.p3]);
      expect([...a.p4]).toEqual([...b.p4]);
      expect([...a.periodicity]).toEqual([...b.periodicity]);
      expect([...a.phase]).toEqual([...b.phase]);
      expect([...a.k]).toEqual([...b.k]);
    });

    it("has the same nonbonded particles, exceptions and settings", () => {
      const a = force<NonbondedForce>(ours!, "NonbondedForce");
      const b = force<NonbondedForce>(theirs!, "NonbondedForce");
      expect([...a.charge]).toEqual([...b.charge]);
      expect([...a.sigma]).toEqual([...b.sigma]);
      expect([...a.epsilon]).toEqual([...b.epsilon]);
      expect(a.exceptionCount).toBe(b.exceptionCount);
      expect([...a.exceptionP1]).toEqual([...b.exceptionP1]);
      expect([...a.exceptionP2]).toEqual([...b.exceptionP2]);
      expect([...a.exceptionChargeProd]).toEqual([...b.exceptionChargeProd]);
      expect([...a.exceptionSigma]).toEqual([...b.exceptionSigma]);
      expect([...a.exceptionEpsilon]).toEqual([...b.exceptionEpsilon]);
      expect({
        method: a.method,
        cutoff: a.cutoff,
        ewaldTolerance: a.ewaldTolerance,
        alpha: a.alpha,
        useDispersionCorrection: a.useDispersionCorrection,
        useSwitchingFunction: a.useSwitchingFunction,
        switchingDistance: a.switchingDistance,
        exceptionsUsePeriodic: a.exceptionsUsePeriodic,
        includeDirectSpace: a.includeDirectSpace,
        recipForceGroup: a.recipForceGroup,
      }).toEqual({
        method: b.method,
        cutoff: b.cutoff,
        ewaldTolerance: b.ewaldTolerance,
        alpha: b.alpha,
        useDispersionCorrection: b.useDispersionCorrection,
        useSwitchingFunction: b.useSwitchingFunction,
        switchingDistance: b.switchingDistance,
        exceptionsUsePeriodic: b.exceptionsUsePeriodic,
        includeDirectSpace: b.includeDirectSpace,
        recipForceGroup: b.recipForceGroup,
      });
    });
  });
}

describe.skipIf(!existsSync(`fixtures/prep/1L2Y.pdb`))("constraining bonds", () => {
  const prepared = prepareSystem(
    readPdb(readFileSync("fixtures/prep/1L2Y.pdb", "utf8")),
    parseForceField(readFileSync(fieldPath, "utf8")),
    parseStandardBonds(readFileSync("vendor/ffxml/residues.xml", "utf8")),
  );

  it("moves hydrogen bonds out of the force and into the constraint list", () => {
    // 🔴 A constrained bond is *removed* from the HarmonicBondForce, not
    // duplicated into both. Keeping both double-counts the restoring force,
    // which is what `flexibleConstraints` exists to opt into and is off by
    // default. So the two counts have to add up to the unconstrained total.
    const free = buildSystem(prepared, { constraints: "none" });
    const rigid = buildSystem(prepared, { constraints: "h-bonds" });
    const total = force<HarmonicBondForce>(free, "HarmonicBondForce").count;
    const kept = force<HarmonicBondForce>(rigid, "HarmonicBondForce").count;
    expect(rigid.constraintCount + kept).toBe(total);
    expect(rigid.constraintCount).toBeGreaterThan(0);

    // Every constraint is a bond with a hydrogen in it, and every bond left in
    // the force has none.
    const masses = rigid.masses;
    for (let i = 0; i < rigid.constraintCount; i++) {
      const a = rigid.constraintP1[i] as number;
      const b = rigid.constraintP2[i] as number;
      expect(Math.min(masses[a] as number, masses[b] as number)).toBeLessThan(1.5);
    }
    const force_ = force<HarmonicBondForce>(rigid, "HarmonicBondForce");
    for (let i = 0; i < force_.count; i++) {
      const a = force_.p1[i] as number;
      const b = force_.p2[i] as number;
      expect(Math.min(masses[a] as number, masses[b] as number)).toBeGreaterThan(1.5);
    }
  });

  it("constrains every bond when asked", () => {
    const rigid = buildSystem(prepared, { constraints: "all-bonds" });
    expect(force<HarmonicBondForce>(rigid, "HarmonicBondForce").count).toBe(0);
    expect(rigid.constraintCount).toBe(prepared.bondTerms.length);
  });
});
