/**
 * "Verify this system": the kernels against the double-precision reference, on
 * whatever the user uploaded.
 *
 * The differential tests in `src/gpu/validate*.ts` compare against **OpenMM**,
 * which is only possible for a fixture whose energies were captured in Python.
 * A user's own system has no such reference, and asking them to produce one
 * would defeat the point of the page. What is available is the engine's other
 * half: `src/reference/` evaluates the same force field in f64 and is held to
 * OpenMM on the committed fixtures, so comparing the device against it says
 * "the kernels agree with the implementation that agrees with OpenMM".
 *
 * That is weaker than a direct comparison and it is not nothing. It catches the
 * failures that depend on the *system* rather than on the code -- a force field
 * with parameters outside the range the kernels were tested on, an atom count
 * that overflows a dispatch, a box that defeats the cell list -- which are
 * exactly the failures a fixture cannot find and a stranger's upload can.
 *
 * It is also slow, and honestly so: the f64 reference evaluates PME on a mesh
 * in TypeScript, seconds for a solvated system. This runs once, before a run,
 * not per step.
 */

import { parseState } from "../engine/state.js";
import {
  parseSystem,
  type NonbondedForce,
  type OpenMMSystem,
} from "../engine/system.js";
import {
  defaultGrid,
  evaluatePotential,
  splittingParameter,
} from "../reference/potential.js";
import { rectangularBox } from "../reference/forces/nonbonded.js";
import { checkMoleculesIntact } from "../engine/wrapping.js";
import { cellHeadroom } from "./cell-list.js";
import { deviceFitness, type DeviceFitness } from "./device.js";
import { StepLoop } from "./loop.js";

export interface ForceAgreement {
  readonly force: string;
  /** kJ/mol, as the reference computes it. */
  readonly referenceEnergy: number;
}

export interface VerifyReport {
  readonly particleCount: number;
  readonly forces: readonly string[];
  readonly box: readonly number[];
  readonly cutoff: number;
  /** nm^-1 and the mesh the engine will use. */
  readonly alpha: number;
  readonly mesh: readonly number[];
  /** kJ/mol, device and reference, over the whole force field. */
  readonly deviceEnergy: number;
  readonly referenceEnergy: number;
  readonly energyRelativeError: number;
  /** Relative RMS difference in the forces. */
  readonly forceRelativeRms: number;
  /** The largest force the reference reports, so the RMS has a scale. */
  readonly maxReferenceForce: number;
  /** Per force type, from the reference, so a user can see what is in there. */
  readonly byForce: Readonly<Record<string, number>>;
  /** Whether the uploaded coordinates have molecules split across the boundary. */
  readonly moleculesIntact: boolean;
  readonly worstBondedSeparation: number;
  /** How much the box can compress before the cell list fails. */
  readonly cellHeadroom: number;
  readonly canRunNpt: boolean;
  /**
   * Whether this system fits this GPU.
   *
   * 🔴 Checked here because here is before the run. The limits that bind are
   * per device, so a system that runs on the machine it was prepared on can
   * exceed a user's — and without this the first sign is an allocation failure
   * part-way through, which reads as a bug rather than as a system too large
   * for the hardware.
   */
  readonly fitness: DeviceFitness;
  readonly thresholds: typeof VERIFY_THRESHOLDS;
  readonly ok: boolean;
  /** Seconds the whole check took, because the user is waiting. */
  readonly elapsed: number;
}

/**
 * Acceptance thresholds.
 *
 * Looser than the fixtures' because this compares the *total* force field in
 * one go rather than group by group, so it carries every f32-versus-f64
 * difference at once -- the fixtures measure 1e-7 to 1e-5 relative per group.
 * What this has to catch is a system the kernels mishandle, which is a
 * percent-scale disagreement or worse, not a part in 10^5.
 */
export const VERIFY_THRESHOLDS = {
  energyRelative: 1e-4,
  forceRelativeRms: 1e-3,
  /** Below this, a barostat will exhaust the cell list. */
  nptHeadroom: 0.05,
} as const;

export async function verifySystem(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
): Promise<VerifyReport> {
  const began = performance.now();
  const system: OpenMMSystem = parseSystem(texts.systemXml);
  const state = parseState(texts.stateXml);
  const box = rectangularBox(state.boxVectors);
  const nonbonded = system.forces.find(
    (f): f is NonbondedForce => f.type === "NonbondedForce",
  );
  if (nonbonded === undefined) {
    throw new Error(
      "this system has no NonbondedForce. The engine implements the forces " +
        "this workflow produces; a system without one is not something it can " +
        "run, and pretending otherwise would give a confident wrong answer.",
    );
  }
  const alpha = splittingParameter(nonbonded);
  const mesh = defaultGrid(nonbonded, box, alpha);

  const loop = await StepLoop.fromParsed(device, system, state, {
    timeStep: 0.002,
    temperature: 300,
    friction: 0,
    constraintTolerance: 1e-6,
    integrator: "verlet",
  });
  let deviceEnergy: number;
  let deviceForces: Float64Array;
  try {
    const measured = await loop.potential(state.positions);
    deviceEnergy = measured.energy;
    deviceForces = measured.forces;
  } finally {
    loop.destroy();
  }

  const reference = evaluatePotential(system, state.positions, state.boxVectors, {
    grid: mesh,
  });

  let squared = 0;
  let referenceSquared = 0;
  let maxReferenceForce = 0;
  for (let i = 0; i < reference.forces.length; i++) {
    const theirs = reference.forces[i] as number;
    squared += ((deviceForces[i] as number) - theirs) ** 2;
    referenceSquared += theirs * theirs;
    maxReferenceForce = Math.max(maxReferenceForce, Math.abs(theirs));
  }
  const forceRelativeRms =
    referenceSquared === 0 ? 0 : Math.sqrt(squared / referenceSquared);
  const energyRelativeError =
    Math.abs(deviceEnergy - reference.energy) / Math.abs(reference.energy);

  const wrapping = checkMoleculesIntact(system, state.positions, box);
  const headroom = cellHeadroom(box, nonbonded.cutoff);
  const fitness = deviceFitness(
    // `device.limits` is a `GPUSupportedLimits`, whose members are accessors
    // rather than own properties, so spreading it yields an empty object.
    Object.fromEntries(
      (
        [
          "maxBufferSize",
          "maxStorageBufferBindingSize",
          "maxComputeWorkgroupStorageSize",
          "maxComputeInvocationsPerWorkgroup",
          "maxComputeWorkgroupSizeX",
          "maxComputeWorkgroupsPerDimension",
          "maxStorageBuffersPerShaderStage",
        ] as const
      ).map((name) => [name, device.limits[name] as number]),
    ),
    system.particleCount,
  );

  return {
    particleCount: system.particleCount,
    forces: system.forces.map((f) => f.type),
    box: Array.from(box as unknown as number[]),
    cutoff: nonbonded.cutoff,
    alpha,
    mesh: Array.from(mesh as unknown as number[]),
    deviceEnergy,
    referenceEnergy: reference.energy,
    energyRelativeError,
    forceRelativeRms,
    maxReferenceForce,
    byForce: reference.byForce,
    moleculesIntact: wrapping.intact,
    worstBondedSeparation: wrapping.worstSeparation,
    cellHeadroom: headroom,
    canRunNpt: headroom >= VERIFY_THRESHOLDS.nptHeadroom,
    fitness,
    thresholds: VERIFY_THRESHOLDS,
    // 🔴 `fitness.fits` is folded in, and it is not redundant even though
    // reaching this line means the buffers were allocated. The memory checks
    // are indeed already proven by getting here; the *other* limits are not.
    // A device reporting fewer than eight storage buffers per stage, or fewer
    // than 64 invocations per workgroup, can still complete this verification
    // -- which binds fewer resources than the force kernels in a run -- and
    // then fail during the run itself. Better to say so now.
    ok:
      energyRelativeError < VERIFY_THRESHOLDS.energyRelative &&
      forceRelativeRms < VERIFY_THRESHOLDS.forceRelativeRms &&
      wrapping.intact &&
      fitness.fits,
    elapsed: (performance.now() - began) / 1000,
  };
}
