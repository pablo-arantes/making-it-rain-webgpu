/**
 * Minimisation on the device, against OpenMM's own.
 *
 * The same comparison the reference gets in test/minimise.slow.test.ts --
 * energies at matched iteration counts, because a minimum is not unique and
 * only a shared path compares tightly -- except that it runs here, where the
 * forces come from the kernels rather than from f64 TypeScript. That makes two
 * things testable that were not:
 *
 * **3HTB.** The f64 reference evaluates PME on a 54^3 mesh on the CPU, which is
 * seconds a call and fifty calls a minimisation. On the device it is
 * milliseconds, so the protein-ligand fixture gets the same parity check as the
 * small one.
 *
 * **Where fixed-point positions stop the search.** Coordinates resolve to 2^-32
 * nm, and a backtracking line search halves its trial step until the energy
 * improves. Below that resolution the device sees no movement, so the search
 * stalls where the reference keeps going. This reports the step it stalled at
 * and the energy it reached, so the limit is a number rather than a worry.
 */

import { parseState } from "../engine/state.js";
import { parseSystem } from "../engine/system.js";
import { maxConstraintError } from "../reference/minimise.js";
import { StepLoop, type RunOptions } from "./loop.js";
import { minimiseOnDevice } from "./minimise.js";

const TIME_STEP = 0.002;
const CONSTRAINT_TOLERANCE = 1e-6;

/**
 * How many iterations the two searches can be expected to agree over.
 *
 * Up to here the path is shared -- L-BFGS's first direction is the gradient and
 * the backtracking is deterministic -- so the energies are a tight comparison
 * of the objective. Beyond it they are not: measured on 1L2Y, the device sits
 * 4.8e-5 from OpenMM at five iterations and 1.9e-3 at twenty, and our own f64
 * reference sits 6.5e-3 at twenty. The device is *closer* to OpenMM there than
 * the reference is, which is the clearest sign available that what grows is
 * divergence between searches and not error in either.
 */
export const PATH_SHARED_ITERATIONS = 5;

/**
 * Acceptance thresholds.
 *
 * The relative energy bound applies only to the shared-path frames. Measured at
 * one, two and five iterations:
 *
 *   1L2Y   9.1e-5   3.5e-5   4.8e-5     (against OpenMM's Reference platform)
 *   3HTB   8.5e-6   2.7e-5   4.3e-4     (against its CPU platform)
 *
 * 3HTB diverges about nine times faster, and that is not a surprise: its
 * direct-space force relative RMS is 1.13e-5 against 1L2Y's 7.65e-7, which is
 * the open item in docs/NONBONDED.md. A line search amplifies a force
 * difference into a different backtracking decision and then a different path.
 * So 1e-3 is about twice the worst measurement, and it is provisional in the
 * specific sense that it should tighten once 3HTB's force error is understood.
 * It would still catch a wrong objective, which is out by orders of magnitude
 * rather than parts in a thousand -- a missing constraint-spring term alone is
 * thousands of kJ/mol.
 *
 * Later frames are held to what can honestly be claimed of them: the energy
 * went down a long way, and the constraints hold to what was asked.
 */
export const MINIMISE_THRESHOLDS = {
  energyRelative: 1e-3,
  constraintError: 2e-6,
  /** kJ/mol per atom a long run must gain over the starting configuration. */
  descentPerAtom: 0.5,
} as const;

export interface MinimiseFrame {
  readonly iterations: number;
  /** Whether this frame is close enough to the start to compare paths. */
  readonly pathShared: boolean;
  readonly energy: number;
  readonly openmmEnergy: number | null;
  readonly energyRelativeError: number | null;
  /** kJ/mol/nm, springs excluded, so it is comparable with OpenMM's forces. */
  readonly forceRms: number;
  readonly openmmForceRms: number | null;
  readonly maxConstraintError: number;
  readonly rounds: number;
  readonly evaluations: number;
  readonly status: string;
  readonly gaveUp: boolean;
  readonly ok: boolean;
}

export interface MinimiseComparison {
  readonly fixture: string;
  readonly particleCount: number;
  /** kJ/mol, evaluated on the device at the fixture's own coordinates. */
  readonly startEnergy: number;
  /** Which OpenMM platform the capture came from, when there is one. */
  readonly openmmPlatform: string | null;
  /** What the fixture's own state violates its constraints by, before anything. */
  readonly startConstraintError: number;
  readonly frames: readonly MinimiseFrame[];
  /** The energy a long run reaches, and whether it is usable afterwards. */
  readonly converged: MinimiseFrame;
  readonly thresholds: typeof MINIMISE_THRESHOLDS;
  readonly ok: boolean;
}

/** What `tools/capture_minimisation.py` writes. */
export interface CapturedMinimisation {
  readonly platform?: string;
  readonly start: {
    readonly energy_kJ_mol: number;
    readonly max_constraint_error: number;
  };
  readonly frames: Record<
    string,
    {
      readonly energy_kJ_mol: number;
      readonly force_rms_per_particle: number;
      readonly max_constraint_error: number;
    }
  >;
}

export async function validateMinimise(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
  captured: CapturedMinimisation | null,
  fixtureName: string,
  iterationCounts: readonly number[] = [1, 2, 5, 20],
  convergedIterations = 200,
): Promise<MinimiseComparison> {
  const system = parseSystem(texts.systemXml);
  const state = parseState(texts.stateXml);
  const options: RunOptions = {
    timeStep: TIME_STEP,
    temperature: 300,
    friction: 0,
    constraintTolerance: CONSTRAINT_TOLERANCE,
    integrator: "verlet",
    cellListInterval: 1,
  };

  // The energy at the fixture's own coordinates, from the device, so that
  // "the energy went down" is measured against something this engine computed
  // rather than against a number from a different platform.
  let startEnergy: number;
  {
    const loop = await StepLoop.fromParsed(device, system, state, options);
    try {
      startEnergy = (await loop.potential(state.positions)).energy;
    } finally {
      loop.destroy();
    }
  }

  const run = async (maxIterations: number): Promise<MinimiseFrame> => {
    // A fresh loop each time: the minimiser starts from the fixture's own
    // coordinates, and reusing one would start from the previous result.
    const loop = await StepLoop.fromParsed(device, system, state, options);
    try {
      const result = await minimiseOnDevice(loop, system, {
        tolerance: 10,
        maxIterations,
        constraintTolerance: CONSTRAINT_TOLERANCE,
      });
      const key = String(maxIterations);
      const expected = captured?.frames[key];
      const energyRelativeError =
        expected === undefined
          ? null
          : Math.abs(result.energy - expected.energy_kJ_mol) /
            Math.abs(expected.energy_kJ_mol);
      const pathShared = maxIterations <= PATH_SHARED_ITERATIONS;
      const constraintError = maxConstraintError(system, result.positions);
      const descended =
        startEnergy - result.energy >
        MINIMISE_THRESHOLDS.descentPerAtom * system.particleCount;
      return {
        iterations: maxIterations,
        pathShared,
        energy: result.energy,
        openmmEnergy: expected?.energy_kJ_mol ?? null,
        energyRelativeError,
        forceRms: result.potentialForceRms,
        openmmForceRms: expected?.force_rms_per_particle ?? null,
        maxConstraintError: constraintError,
        rounds: result.rounds,
        evaluations: result.evaluations,
        status: result.status,
        gaveUp: result.gaveUp,
        ok:
          constraintError < MINIMISE_THRESHOLDS.constraintError &&
          !result.gaveUp &&
          // A shared-path frame is compared against OpenMM; a later one is held
          // to having descended, because by then the searches are in different
          // places and an energy comparison would be measuring that.
          (pathShared && energyRelativeError !== null
            ? energyRelativeError < MINIMISE_THRESHOLDS.energyRelative
            : descended || pathShared),
      };
    } finally {
      loop.destroy();
    }
  };

  const frames: MinimiseFrame[] = [];
  for (const count of iterationCounts) frames.push(await run(count));
  const converged = await run(convergedIterations);

  return {
    fixture: fixtureName,
    particleCount: system.particleCount,
    startEnergy,
    openmmPlatform: captured?.platform ?? null,
    startConstraintError: maxConstraintError(system, state.positions),
    frames,
    converged,
    thresholds: MINIMISE_THRESHOLDS,
    ok: frames.every((frame) => frame.ok) && converged.ok,
  };
}
