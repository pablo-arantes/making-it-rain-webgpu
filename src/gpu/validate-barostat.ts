/**
 * The barostat on the device.
 *
 * The move itself is tested in f64 (test/barostat.test.ts) and the molecule
 * decomposition against OpenMM (test/molecules.test.ts). What is left, and what
 * can only be checked here, is that **the box is genuinely dynamic**: that
 * changing it rewrites every uniform that carries a box-derived quantity, and
 * not merely most of them.
 *
 * That is a long list -- the pair and exception kernels' boxes in f32 *and* in
 * fixed point, three reciprocal-space blocks whose `prefactor` carries
 * 1/volume, the cell geometry, each restraint's wrap, and two host-side energy
 * terms -- and the failure mode of missing one is not a crash. It is an energy
 * that is slightly wrong in a way that looks like ordinary single-precision
 * disagreement, and a barostat that then samples the wrong ensemble.
 *
 * The check is device against device, and asserted **exactly**. `setBox` has to
 * leave the loop in the state a loop *constructed* at that box would be in --
 * the two reach every uniform by different routes, construction through each
 * `prepare*` and `setBox` through each `pack*`, so agreement means both routes
 * wrote the same thing. There is no tolerance to choose: the same kernels on
 * the same data give the same bits, and a stale uniform is not a small
 * difference anyway. The Ewald prefactor alone is 2% off for a 2% volume
 * change, which is thousands of kJ/mol.
 *
 * It used to compare against the f64 reference at the new box instead. That
 * works and it cost minutes per fixture, because `evaluatePotential` runs a
 * 54^3 mesh in TypeScript. It is also the weaker test: the device-versus-f64
 * comparison at the *starting* box already exists
 * (src/gpu/validate-restraint.ts), so what was missing was only whether a box
 * change reproduces it -- which is exactly what construction gives for free.
 */

import { parseState } from "../engine/state.js";
import {
  parseSystem,
  type MonteCarloBarostat,
  type OpenMMSystem,
} from "../engine/system.js";
import { findMolecules } from "../engine/molecules.js";
import { boxLengths, scaleBox, scaleMolecules } from "../reference/barostat.js";
import { maxConstraintError } from "../reference/minimise.js";
import { StepLoop, type RunOptions } from "./loop.js";

/** Thresholds. */
export const BAROSTAT_THRESHOLDS = {
  /**
   * Zero. The two routes run the same kernels over the same data, so they agree
   * bit for bit or one of them wrote a different uniform.
   */
  energyRelative: 0,
  /**
   * How much the worst constraint violation may change under scaling.
   *
   * Rigid-molecule scaling translates each molecule, so in exact arithmetic
   * this is zero; what is left is f64 round-off on coordinates of a few
   * nanometres, which is why the bound is 1e-12 rather than a tolerance.
   */
  constraintErrorChange: 1e-12,
} as const;

export interface BoxChange {
  readonly lengthScale: number;
  /** From a loop whose box was changed by `setBox`. */
  readonly afterSetBox: number;
  /** From a loop constructed at that box, which reaches the uniforms another way. */
  readonly afterConstruction: number;
  readonly difference: number;
  /**
   * How much rigid-molecule scaling changed the worst constraint violation.
   *
   * The *change*, not the value. This fixture's own state violates its
   * constraints by 25% before anything runs, so an absolute bound here would
   * measure the fixture rather than the move -- which is what the first version
   * of this check did. What the move claims is that internal geometry is
   * untouched, so the violation must come out exactly as it went in.
   */
  readonly constraintErrorChange: number;
  readonly ok: boolean;
}

export interface BarostatComparison {
  readonly particleCount: number;
  readonly molecules: number;
  readonly startVolume: number;
  /** The energy agreement at each box the test moved to. */
  readonly changes: readonly BoxChange[];
  readonly steps: number;
  readonly attempts: number;
  readonly accepted: number;
  /** nm^3, after the run. */
  readonly finalVolume: number;
  readonly volumeScale: number;
  readonly finite: boolean;
  readonly thresholds: typeof BAROSTAT_THRESHOLDS;
  readonly ok: boolean;
}

/** A system with a barostat attached, as a run would attach one. */
function withBarostat(system: OpenMMSystem, frequency: number): OpenMMSystem {
  const barostat: MonteCarloBarostat = {
    type: "MonteCarloBarostat",
    name: "MonteCarloBarostat",
    forceGroup: 0,
    defaultPressure: 1,
    defaultTemperature: 300,
    frequency,
    randomSeed: 0,
  };
  return { ...system, forces: [...system.forces, barostat] };
}

export async function validateBarostat(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
  /**
   * An equilibrated configuration to run from.
   *
   * 🔴 Not `state.xml`. That state is un-minimised -- 5.6e5 kJ/mol/nm on 3HTB
   * -- and a hundred steps of NVE from it diverges: the first attempt at this
   * reported "2.15e9 is outside the fixed-point range" for a *position*, which
   * is a blown-up simulation and not a barostat defect. A move that is
   * evaluated on a diverging trajectory tells you nothing about the move.
   */
  start?: { readonly positions: Float64Array; readonly velocities: Float64Array },
  steps = 100,
): Promise<BarostatComparison> {
  const parsed = parseSystem(texts.systemXml);
  const parsedState = parseState(texts.stateXml);
  const state =
    start === undefined
      ? parsedState
      : { ...parsedState, positions: start.positions, velocities: start.velocities };
  const molecules = findMolecules(parsed);
  const system = withBarostat(parsed, 25);
  const options: RunOptions = {
    timeStep: 0.002,
    temperature: 300,
    friction: 0,
    constraintTolerance: 1e-6,
    integrator: "verlet",
    seed: 0x5eed,
  };

  // What the fixture already violates its constraints by, so the move can be
  // measured against it rather than against zero.
  const startingConstraintError = maxConstraintError(system, state.positions);

  // The mesh the starting box implies, which a run keeps for its whole life.
  const probe = await StepLoop.fromParsed(device, system, state, options);
  const startingMesh = probe.mesh;
  probe.destroy();

  const changes: BoxChange[] = [];
  // Both directions, and larger than a barostat move, so a stale uniform
  // cannot hide inside the agreement.
  for (const lengthScale of [1.02, 0.98]) {
    const trialBox = scaleBox(state.boxVectors, lengthScale);
    const trialLengths = boxLengths(trialBox);
    const trialPositions = Float64Array.from(state.positions);
    scaleMolecules(trialPositions, molecules, lengthScale);

    const changed = await StepLoop.fromParsed(device, system, state, options);
    let afterSetBox: number;
    try {
      changed.setBox(trialLengths);
      afterSetBox = (await changed.potential(trialPositions)).energy;
    } finally {
      changed.destroy();
    }

    // Constructed at the trial box, but with the *starting* box's mesh, because
    // a run keeps its mesh when the box moves and that is what `setBox`
    // preserves. Without pinning it the two would differ for a legitimate
    // reason and the comparison would mean nothing.
    const constructed = await StepLoop.fromParsed(
      device,
      system,
      { ...state, boxVectors: trialBox },
      { ...options, mesh: startingMesh },
    );
    let afterConstruction: number;
    try {
      afterConstruction = (await constructed.potential(trialPositions)).energy;
    } finally {
      constructed.destroy();
    }

    const difference = Math.abs(afterSetBox - afterConstruction);
    const constraintErrorChange = Math.abs(
      maxConstraintError(system, trialPositions) - startingConstraintError,
    );
    changes.push({
      lengthScale,
      afterSetBox,
      afterConstruction,
      difference,
      constraintErrorChange,
      ok:
        difference <= BAROSTAT_THRESHOLDS.energyRelative &&
        constraintErrorChange < BAROSTAT_THRESHOLDS.constraintErrorChange,
    });
  }

  // And a run with the barostat live.
  const loop = await StepLoop.fromParsed(device, system, state, options);
  let attempts = 0;
  let accepted = 0;
  let finalVolume = 0;
  let volumeScale = 0;
  let finite = false;
  try {
    await loop.run(steps);
    const statistics = loop.barostatStatistics;
    attempts = statistics?.attempts ?? 0;
    accepted = statistics?.accepted ?? 0;
    volumeScale = statistics?.volumeScale ?? 0;
    const box = loop.boxLengths;
    finalVolume = (box[0] as number) * (box[1] as number) * (box[2] as number);
    const positions = await loop.positions();
    finite = positions.every((value) => Number.isFinite(value));
  } finally {
    loop.destroy();
  }

  const startVolume =
    (state.boxVectors[0][0] as number) *
    (state.boxVectors[1][1] as number) *
    (state.boxVectors[2][2] as number);

  return {
    particleCount: system.particleCount,
    molecules: molecules.count,
    startVolume,
    changes,
    steps,
    attempts,
    accepted,
    finalVolume,
    volumeScale,
    finite,
    thresholds: BAROSTAT_THRESHOLDS,
    ok:
      changes.every((change) => change.ok) &&
      finite &&
      // 100 steps at frequency 25 is four attempts; if none happened the
      // barostat is not wired into the loop at all.
      attempts === Math.floor(steps / 25),
  };
}
