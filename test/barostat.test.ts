/**
 * `MonteCarloBarostat`: the move's defining properties, and its bookkeeping.
 *
 * The energy side of a barostat is the force field's, which is tested
 * elsewhere. What is specific to the barostat is the *move*: what it scales,
 * what it must leave alone, when it fires, and how it decides. Those are all
 * checkable without a force field, which is what this does -- a stub potential
 * makes the Metropolis arithmetic exact instead of statistical.
 */

import { describe, expect, it } from "vitest";
import {
  BAR_TO_KJ_PER_MOL_NM3,
  barostatActsNow,
  barostatAttempt,
  boxVolume,
  initialBarostatState,
  scaleBox,
  scaleMolecules,
  type BarostatState,
} from "../src/reference/barostat.js";
import { findMolecules } from "../src/engine/molecules.js";
import { BOLTZMANN_KJ_PER_MOL_K } from "../src/engine/units.js";
import type { MonteCarloBarostat, OpenMMSystem, Vec3 } from "../src/engine/system.js";
import { fixture, FIXTURE_NAMES } from "./fixtures.js";

const box: [Vec3, Vec3, Vec3] = [
  [3, 0, 0],
  [0, 4, 0],
  [0, 0, 5],
];

const barostat: MonteCarloBarostat = {
  type: "MonteCarloBarostat",
  name: "MonteCarloBarostat",
  forceGroup: 0,
  defaultPressure: 1,
  defaultTemperature: 300,
  frequency: 25,
  randomSeed: 0,
};

describe("the proposal schedule", () => {
  it("fires on the frequency-th call, not the zeroth", () => {
    // The opposite of CMMotionRemover, which acts on step zero. The two sit a
    // few lines apart in a step loop, which is why this is pinned.
    const state = initialBarostatState(box);
    const fired: number[] = [];
    for (let call = 1; call <= 75; call++) {
      if (barostatActsNow(state, 25)) fired.push(call);
    }
    expect(fired).toEqual([25, 50, 75]);
  });

  it("never fires at frequency zero", () => {
    const state = initialBarostatState(box);
    for (let call = 0; call < 100; call++) {
      expect(barostatActsNow(state, 0)).toBe(false);
    }
  });

  it("starts with a proposal width of one percent of the volume", () => {
    expect(initialBarostatState(box).volumeScale).toBeCloseTo(0.01 * 60, 12);
    expect(boxVolume(box)).toBe(60);
  });
});

describe("pressure units", () => {
  it("converts a bar to kJ/mol/nm^3", () => {
    // A bar is 1e5 J/m^3; a nm^3 is 1e-27 m^3; a mole is Avogadro's number.
    expect(BAR_TO_KJ_PER_MOL_NM3).toBeCloseTo(6.02214076e23 * 1e5 * 1e-27 / 1000, 12);
  });
});

describe("scaleMolecules", () => {
  for (const name of FIXTURE_NAMES) {
    const f = fixture(name);
    describe.skipIf(f === null)(name, () => {
      it("leaves every intramolecular distance untouched", () => {
        // The defining property. Scaling atoms individually instead would
        // stretch every bond by the factor, violating a rigid water's
        // constraints by 0.3% on a 1% volume move.
        const molecules = findMolecules(f!.system);
        const before = Float64Array.from(f!.state.positions);
        const after = Float64Array.from(before);
        scaleMolecules(after, molecules, 1.01);
        let worst = 0;
        for (let molecule = 0; molecule < molecules.count; molecule++) {
          const begin = molecules.starts[molecule] as number;
          const end = molecules.starts[molecule + 1] as number;
          for (let slot = begin; slot + 1 < end; slot++) {
            const a = molecules.atoms[slot] as number;
            const b = molecules.atoms[slot + 1] as number;
            const separation = (p: Float64Array): number =>
              Math.hypot(
                (p[b * 3] as number) - (p[a * 3] as number),
                (p[b * 3 + 1] as number) - (p[a * 3 + 1] as number),
                (p[b * 3 + 2] as number) - (p[a * 3 + 2] as number),
              );
            worst = Math.max(worst, Math.abs(separation(after) - separation(before)));
          }
        }
        // A rigid translation in f64, so this is round-off on coordinates of a
        // few nanometres and nothing else.
        expect(worst).toBeLessThan(1e-12);
      });

      it("scales each molecule's centroid by exactly the factor", () => {
        const molecules = findMolecules(f!.system);
        const scale = 1.0 / 1.01;
        const before = Float64Array.from(f!.state.positions);
        const after = Float64Array.from(before);
        scaleMolecules(after, molecules, scale);
        const centroid = (p: Float64Array, molecule: number, axis: number): number => {
          const begin = molecules.starts[molecule] as number;
          const end = molecules.starts[molecule + 1] as number;
          let sum = 0;
          for (let slot = begin; slot < end; slot++) {
            sum += p[(molecules.atoms[slot] as number) * 3 + axis] as number;
          }
          return sum / (end - begin);
        };
        let worstRelative = 0;
        for (let molecule = 0; molecule < molecules.count; molecule++) {
          for (let axis = 0; axis < 3; axis++) {
            const wanted = scale * centroid(before, molecule, axis);
            const got = centroid(after, molecule, axis);
            worstRelative = Math.max(
              worstRelative,
              Math.abs(got - wanted) / Math.max(1e-6, Math.abs(wanted)),
            );
          }
        }
        expect(worstRelative).toBeLessThan(1e-10);
      });
    });
  }

  it("uses the unweighted centroid, not the centre of mass", () => {
    // OpenMM's choice, and it is visible: a water's centre of mass sits almost
    // on its oxygen while its centroid does not, so the two give different
    // offsets for the same scale factor.
    const system = {
      particleCount: 3,
      masses: Float64Array.from([16, 1, 1]),
      constraintCount: 0,
      constraintP1: new Int32Array(0),
      constraintP2: new Int32Array(0),
      constraintDistance: new Float64Array(0),
      forces: [],
      boxVectors: box,
      openmmVersion: "",
    } as unknown as OpenMMSystem;
    // One molecule, formed by nothing: with no bonded terms every atom is its
    // own molecule, so this is built by hand.
    const molecules = {
      moleculeOf: Int32Array.from([0, 0, 0]),
      count: 1,
      starts: Int32Array.from([0, 3]),
      atoms: Int32Array.from([0, 1, 2]),
      largest: 3,
    };
    const positions = Float64Array.from([0, 0, 0, 3, 0, 0, 0, 3, 0]);
    scaleMolecules(positions, molecules, 2);
    // Centroid is (1, 1, 0); doubling it gives (2, 2, 0), an offset of (1, 1, 0).
    expect(Array.from(positions)).toEqual([1, 1, 0, 4, 1, 0, 1, 4, 0]);
    void system;
  });
});

describe("scaleBox", () => {
  it("scales every component", () => {
    expect(scaleBox(box, 2)).toEqual([
      [6, 0, 0],
      [0, 8, 0],
      [0, 0, 10],
    ]);
  });
});

describe("the Metropolis decision", () => {
  const tinySystem = {
    particleCount: 2,
    masses: Float64Array.from([1, 1]),
    constraintCount: 0,
    constraintP1: new Int32Array(0),
    constraintP2: new Int32Array(0),
    constraintDistance: new Float64Array(0),
    forces: [],
    boxVectors: box,
    openmmVersion: "",
  } as unknown as OpenMMSystem;
  const molecules = {
    moleculeOf: Int32Array.from([0, 1]),
    count: 2,
    starts: Int32Array.from([0, 1, 2]),
    atoms: Int32Array.from([0, 1]),
    largest: 1,
  };
  const positions = Float64Array.from([1, 1, 1, 2, 2, 2]);

  /** A potential that reports a fixed value for the second call. */
  const stub = (initial: number, final: number) => {
    let calls = 0;
    return async (): Promise<number> => {
      calls++;
      return calls === 1 ? initial : final;
    };
  };

  const draws = (...values: number[]): (() => number) => {
    let i = 0;
    return () => {
      const value = values[i] ?? 0.5;
      i++;
      return value;
    };
  };

  it("accepts a downhill move without drawing a second deviate", async () => {
    // It matters that the deviate is not drawn: the stream then depends on the
    // trajectory, so reproducing a run needs the same seed *and* the same
    // sequence of accepted moves.
    const state: BarostatState = {
      step: 0,
      volumeScale: 1,
      attempted: 0,
      accepted: 0,
    };
    let drawn = 0;
    const uniform = (): number => {
      drawn++;
      return 0.75;
    };
    const result = await barostatAttempt(
      tinySystem,
      barostat,
      state,
      box,
      positions,
      stub(0, -1000),
      uniform,
      molecules,
    );
    expect(result.accepted).toBe(true);
    expect(drawn).toBe(1);
    expect(result.work).toBeLessThan(0);
  });

  it("rejects when the deviate exceeds exp(-w/kT), and keeps the old state", async () => {
    const state: BarostatState = {
      step: 0,
      volumeScale: 1,
      attempted: 0,
      accepted: 0,
    };
    const kT = BOLTZMANN_KJ_PER_MOL_K * 300;
    // Energy uphill by 10 kT: exp(-10) = 4.5e-5, so 0.5 rejects.
    const result = await barostatAttempt(
      tinySystem,
      barostat,
      state,
      box,
      positions,
      stub(0, 10 * kT),
      draws(0.75, 0.5),
      molecules,
    );
    expect(result.accepted).toBe(false);
    expect(result.boxVectors).toEqual(box);
    expect(Array.from(result.positions)).toEqual(Array.from(positions));
    expect(state.accepted).toBe(0);
    expect(state.attempted).toBe(1);
  });

  it("includes the pressure-volume and molecule-count terms", async () => {
    const state: BarostatState = {
      step: 0,
      volumeScale: 1,
      attempted: 0,
      accepted: 0,
    };
    // A deviate of 0.75 gives deltaVolume = 1 * 2 * 0.25 = +0.5 nm^3.
    const result = await barostatAttempt(
      tinySystem,
      { ...barostat, defaultPressure: 1000 },
      state,
      box,
      positions,
      stub(0, 0),
      draws(0.75, 0.0),
      molecules,
    );
    expect(result.deltaVolume).toBeCloseTo(0.5, 12);
    expect(result.lengthScale).toBeCloseTo(Math.cbrt(60.5 / 60), 12);
    const kT = BOLTZMANN_KJ_PER_MOL_K * 300;
    const expected =
      1000 * BAR_TO_KJ_PER_MOL_NM3 * 0.5 - 2 * kT * Math.log(60.5 / 60);
    expect(result.work).toBeCloseTo(expected, 10);
  });

  it("counts molecules rather than atoms in the logarithm", async () => {
    // Two one-atom molecules against one two-atom molecule, same atoms. The
    // logarithm's coefficient has to follow the molecules.
    const oneMolecule = {
      moleculeOf: Int32Array.from([0, 0]),
      count: 1,
      starts: Int32Array.from([0, 2]),
      atoms: Int32Array.from([0, 1]),
      largest: 2,
    };
    const run = async (which: typeof molecules): Promise<number> => {
      const state: BarostatState = {
        step: 0,
        volumeScale: 1,
        attempted: 0,
        accepted: 0,
      };
      const result = await barostatAttempt(
        tinySystem,
        barostat,
        state,
        box,
        positions,
        stub(0, 0),
        draws(0.75, 0),
        which,
      );
      return result.work;
    };
    const kT = BOLTZMANN_KJ_PER_MOL_K * 300;
    const difference = (await run(molecules)) - (await run(oneMolecule));
    expect(difference).toBeCloseTo(-kT * Math.log(60.5 / 60), 12);
  });
});

describe("adapting the proposal width", () => {
  const makeState = (attempted: number, accepted: number): BarostatState => ({
    step: 0,
    volumeScale: 1,
    attempted,
    accepted,
  });

  it("shrinks after ten attempts with fewer than a quarter accepted", async () => {
    const state = makeState(9, 1);
    const kT = BOLTZMANN_KJ_PER_MOL_K * 300;
    await barostatAttempt(
      { particleCount: 0, forces: [], masses: new Float64Array(0) } as unknown as OpenMMSystem,
      barostat,
      state,
      box,
      new Float64Array(0),
      async () => 100 * kT,
      () => 0.9,
      { moleculeOf: new Int32Array(0), count: 0, starts: Int32Array.from([0]), atoms: new Int32Array(0), largest: 0 },
    );
    expect(state.volumeScale).toBeCloseTo(1 / 1.1, 12);
    expect(state.attempted).toBe(0);
  });

  it("grows after ten attempts with more than three quarters accepted, capped", async () => {
    const state = makeState(9, 9);
    await barostatAttempt(
      { particleCount: 0, forces: [], masses: new Float64Array(0) } as unknown as OpenMMSystem,
      barostat,
      state,
      box,
      new Float64Array(0),
      async () => -1,
      () => 0.5,
      { moleculeOf: new Int32Array(0), count: 0, starts: Int32Array.from([0]), atoms: new Int32Array(0), largest: 0 },
    );
    expect(state.volumeScale).toBeCloseTo(1.1, 12);

    // And the cap: 30% of the volume, which is 18 for this box.
    const wide = makeState(9, 9);
    wide.volumeScale = 100;
    await barostatAttempt(
      { particleCount: 0, forces: [], masses: new Float64Array(0) } as unknown as OpenMMSystem,
      barostat,
      wide,
      box,
      new Float64Array(0),
      async () => -1,
      () => 0.5,
      { moleculeOf: new Int32Array(0), count: 0, starts: Int32Array.from([0]), atoms: new Int32Array(0), largest: 0 },
    );
    expect(wide.volumeScale).toBeCloseTo(18, 12);
  });
});
