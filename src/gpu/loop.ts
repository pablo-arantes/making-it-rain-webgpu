/**
 * The step loop: resident buffers, cached pipelines, one submission per batch.
 *
 * Everything else in src/gpu evaluates one configuration -- a force at a set of
 * coordinates, a step from a captured frame -- and waits for the answer. This
 * runs dynamics, which is a different problem: the same forty-odd kernels every
 * step, nothing to read back until a reporter asks, and no host arithmetic on
 * coordinates anywhere in between.
 *
 * Three properties it is built for, in the order they cost performance if lost.
 *
 * **Nothing is allocated per step.** Every buffer comes from the pool and lives
 * as long as the run; every pipeline is compiled once; every bind group is
 * built once. A step is a list of passes encoded into an existing encoder.
 *
 * **Nothing is read back per step.** A readback drains the pipeline, so one per
 * step would dominate everything else. The cell list is rebuilt on the device,
 * the centre-of-mass momentum is reduced on the device, and the only transfers
 * are at a reporting interval the caller chooses.
 *
 * **Nothing is uploaded per step.** The force accumulators are cleared by a
 * dispatch rather than a host copy, and the one quantity that changes from step
 * to step -- the generator's step counter -- travels as a dynamic offset into a
 * ring of parameter blocks written once per batch.
 *
 * What it is *not* built for is independence from the single-configuration
 * paths. It calls the same `prepare*` and `plan*` functions the differential
 * tests do, because a loop with its own force code would be a loop whose forces
 * are untested.
 */

import { parseState, type OpenMMState } from "../engine/state.js";
import {
  parseSystem,
  type NonbondedForce,
  type OpenMMSystem,
  type PositionRestraintForce,
  type Vec3,
} from "../engine/system.js";
import { ewaldAlpha, pmeMinimumGridSize, smoothFftSize } from "../engine/units.js";
import { splitConstraints } from "../engine/constraint-clusters.js";
import {
  dispersionCorrection,
  netChargeCorrection,
  rectangularBox,
  selfEnergy,
} from "../reference/forces/nonbonded.js";
import { centreOfMassRemover, removesOnStep } from "../reference/com-motion.js";
import {
  barostatActsAfterSteps,
  barostatAttempt,
  initialBarostatState,
  type BarostatState,
} from "../reference/barostat.js";
import { findMolecules, type Molecules } from "../engine/molecules.js";
import {
  threefry2x32,
  openUnitInterval,
  BAROSTAT_DOMAIN,
} from "../reference/random.js";
import {
  degreesOfFreedom as countDegreesOfFreedom,
  kineticEnergy,
  shiftedKineticEnergy,
} from "../reference/kinetic.js";
import { fromFixed, fromWords } from "../reference/fixed-point.js";
import { BufferPool } from "./allocator.js";
import { pipelineCacheFor, readBuffer } from "./compute.js";
import { profilePasses, type Profile } from "./profile.js";
import {
  encodePasses,
  withOffsets,
  type Pass,
  type PipelineCache,
} from "./pipeline.js";
import { readableStorageUsage, uniformUsage } from "./usage.js";
import { packPositions, planBonded, prepareBonded, unpackFixed, type BondedForce } from "./forces.js";
import {
  packExceptionParams,
  packPairParams,
  planDirect,
  prepareDirect,
  type Accumulators,
  type ResidentDirect,
} from "./nonbonded.js";
import { planCellList, planWrapPositions, updateCellBox } from "./cell-list.js";
import {
  packConvolveParams,
  packInterpolateParams,
  packPmeParams,
  planPme,
  preparePme,
  type ResidentPme,
} from "./pme.js";
import { planZero } from "./zero.js";
import {
  packRestraintParams,
  planRestraint,
  prepareRestraint,
  type ResidentRestraint,
} from "./restraint.js";
import {
  packConstraints,
  planConstraints,
  uploadConstraintTables,
} from "./constrain.js";
import {
  planPhase,
  prepareIntegrator,
  writeStepParamsRing,
  STEP_PARAMS_STRIDE,
  type IntegratorBuffers,
  type IntegratorOptions,
} from "./integrate.js";
import { REDUCE } from "./kernels.js";

export interface RunOptions {
  /** ps. */
  readonly timeStep: number;
  /** K. */
  readonly temperature: number;
  /** ps^-1. Zero makes the run deterministic without a seed. */
  readonly friction: number;
  readonly constraintTolerance: number;
  /** Required when there is friction: a run records its seed. */
  readonly seed?: number;
  readonly integrator: "verlet" | "lfmiddle";
  /**
   * How far past the cutoff to build the neighbour list, nm.
   *
   * `DEFAULT_VERLET_SKIN` by default; 0 builds no list and sends every atom
   * through the 27-cell shell scan, which is what the differential tests use.
   *
   * 🔴 This replaced `cellListInterval`, and the reason is worth keeping. That
   * option rebuilt the cell list every N steps, and the pair kernel scanned the
   * 27 cells around an atom's *recorded* home cell -- so a pair inside the
   * cutoff was only guaranteed to be found while every atom had moved less than
   * `cellSize - cutoff`, which is whatever the box happens to leave: 0.087 nm
   * for 3HTB's 6.52 nm box, and arbitrarily close to zero for another box.
   * Losing a pair is not a rounding error, it is a discontinuity in the force:
   * measured on 3HTB at an interval of 20, 13.7 kJ/mol per atom per ns against
   * 0.0023 at every step.
   *
   * A skin is the fix, and it is a different shape from an interval. The list
   * holds everything within `cutoff + skin`, so it stays correct while no atom
   * has moved `skin / 2` -- and *that* is tested on the device every step
   * rather than assumed. The rebuild happens when the geometry says so, not on
   * a schedule, so there is no value of this that silently loses a pair.
   */
  readonly verletSkin?: number;
  /**
   * The largest number of steps encoded into one submission.
   *
   * It sets the size of the parameter ring, which is allocated once because the
   * phase bind groups reference it. A longer `run` is chunked. Larger batches
   * mean fewer synchronisations and a longer wait before a run can be stopped.
   */
  readonly maxBatch?: number;
  /**
   * The PME mesh, overriding the one the starting box implies.
   *
   * A run keeps its mesh for its whole life, as OpenMM does, so a comparison
   * between a loop whose box was *changed* and one *constructed* at that box
   * has to pin the mesh or the two differ for a legitimate reason. See
   * src/gpu/validate-barostat.ts.
   */
  readonly mesh?: Vec3;
}

/** Steps per submission, unless the caller says otherwise. */
export const DEFAULT_MAX_BATCH = 100;

export interface Report {
  readonly step: number;
  /** ps. */
  readonly time: number;
  /** kJ/mol, including the host-side corrections the kernels do not compute. */
  readonly potentialEnergy: number;
  /**
   * kJ/mol, from the velocities as they stand.
   *
   * This is what `LangevinIntegrator` reports and so what the notebook logs.
   */
  readonly kineticEnergy: number;
  /**
   * kJ/mol, from velocities shifted half a step forward and re-projected.
   *
   * This is `VerletIntegrator`'s convention, and it is the one a conserved
   * quantity needs: with the unshifted energy the total oscillates at O(dt)
   * rather than drifting, which makes an energy-drift measurement meaningless.
   * Measured on the captured 1L2Y trajectory, the same eight steps give a total
   * change of -1.25 kJ/mol shifted and -55.3 unshifted. See docs/DYNAMICS.md.
   */
  readonly shiftedKineticEnergy: number;
  /** Potential plus the shifted kinetic energy: the quantity Verlet conserves. */
  readonly totalEnergy: number;
  /** K, from the unshifted kinetic energy and the constrained degrees of freedom. */
  readonly temperature: number;
}

/** The constant terms the kernels leave to the host. */
interface EnergyOffsets {
  readonly dispersion: number;
  readonly self: number;
  readonly netCharge: number;
}

export class StepLoop {
  private stepCount = 0;
  private barostatState: BarostatState | undefined;
  private molecules: Molecules | undefined;
  /**
   * The barostat's deviate counter.
   *
   * A barostat draws one or two uniforms per attempt -- the second only when
   * the move is uphill -- so the count is not a function of the step number.
   * It is carried here, and `run.json` has to record it alongside the seed for
   * a run to be resumable.
   */
  private barostatDraw = 0;

  private constructor(
    readonly device: GPUDevice,
    readonly pool: BufferPool,
    readonly cache: PipelineCache,
    readonly system: OpenMMSystem,
    readonly options: RunOptions,
    private readonly accumulators: Accumulators,
    private readonly buffers: IntegratorBuffers,
    private offsets: EnergyOffsets,
    /**
     * The cell, as vectors rather than three edge lengths.
     *
     * 🔴 It held `Vec3` and the barostat flattened its trial vectors into it
     * with `boxLengths`, which silently assumes a rectangular cell: the
     * off-diagonals were discarded on every rescale. Holding the vectors is
     * what lets a skewed cell be rescaled at all.
     */
    private boxVectors: readonly [Vec3, Vec3, Vec3],
    private readonly alpha: number,
    private readonly nonbonded: NonbondedForce,
    private readonly direct: ResidentDirect,
    private readonly pme: ResidentPme,
    private readonly restraints: readonly ResidentRestraint[],
    private readonly barostat:
      | import("../engine/system.js").MonteCarloBarostat
      | undefined,
    private readonly degreesOfFreedom: number,
    private readonly comFrequency: number,
    /**
     * Folding every coordinate into the box.
     *
     * Separate from `cellPasses` only because the one-shot paths build the
     * cell list without it. The pair kernels difference this pass's output, so
     * it runs every step.
     */
    private readonly wrapPass: Pass,
    /** Rebuilding the cell list. */
    private readonly cellPasses: readonly Pass[],
    /** Clearing the accumulators, then every force. */
    private readonly forcePasses: readonly Pass[],
    /** Momentum reduction and subtraction. */
    private readonly comPasses: readonly Pass[],
    /** The integrator's phases, with the constraint solves interleaved. */
    private readonly phaseBuilder: (offset: number) => readonly Pass[],
  ) {}

  static async create(
    device: GPUDevice,
    texts: { readonly systemXml: string; readonly stateXml: string },
    options: RunOptions,
  ): Promise<StepLoop> {
    const system = parseSystem(texts.systemXml);
    const state = parseState(texts.stateXml);
    return StepLoop.fromParsed(device, system, state, options);
  }

  static async fromParsed(
    device: GPUDevice,
    system: OpenMMSystem,
    state: OpenMMState,
    options: RunOptions,
  ): Promise<StepLoop> {
    const pool = new BufferPool(device);
    const cache = pipelineCacheFor(device);
    const n = system.particleCount;
    const components = n * 3;

    const nonbonded = system.forces.find(
      (f): f is NonbondedForce => f.type === "NonbondedForce",
    );
    if (nonbonded === undefined) {
      throw new Error("a run needs a NonbondedForce; this system has none");
    }
    const box = rectangularBox(state.boxVectors);
    // An uploaded system may pin alpha; otherwise it follows from the cutoff
    // and the tolerance, exactly as `NonbondedForce` does it.
    const alpha =
      nonbonded.alpha !== 0
        ? nonbonded.alpha
        : ewaldAlpha(nonbonded.cutoff, nonbonded.ewaldTolerance);
    const grid =
      options.mesh ??
      (box.map((extent) =>
        smoothFftSize(pmeMinimumGridSize(alpha, extent, nonbonded.ewaldTolerance)),
      ) as unknown as [number, number, number]);

    // The integrator's buffers double as the shared accumulators: `positions`
    // is what every force kernel reads and what the integrator advances, so
    // there is one copy of the coordinates and nothing to keep in step.
    const constraintSplit = splitConstraints(system);
    const tables = packConstraints(system, constraintSplit);
    uploadConstraintTables(pool, tables);
    const buffers = prepareIntegrator(pool, system);
    const accumulators: Accumulators = {
      positions: buffers.positions,
      forces: buffers.forces,
      energy: pool.buffer({
        label: "loop/energy",
        byteLength: 8,
        usage: readableStorageUsage(),
      }),
    };

    device.queue.writeBuffer(
      buffers.positions,
      0,
      packPositions(state.positions) as unknown as BufferSource,
    );
    // A `State` need not carry velocities; a run that starts without them
    // starts from rest, which is what OpenMM does until they are set.
    device.queue.writeBuffer(
      buffers.velocities,
      0,
      (state.velocities === undefined
        ? new Float32Array(components)
        : Float32Array.from(state.velocities)) as unknown as BufferSource,
    );
    device.queue.writeBuffer(buffers.residual, 0, new Float32Array(components));

    const direct = prepareDirect(
      pool,
      nonbonded,
      state.boxVectors,
      alpha,
      options.verletSkin,
    );
    const pme = preparePme(pool, nonbonded, state.boxVectors, alpha, grid);
    const bonded = system.forces.filter(
      (f): f is BondedForce =>
        f.type === "HarmonicBondForce" ||
        f.type === "HarmonicAngleForce" ||
        f.type === "PeriodicTorsionForce",
    );

    const wrapPass = await planWrapPositions(cache, direct.cellList, buffers.positions);
    const cellPasses = await planCellList(cache, direct.cellList, buffers.positions);

    const forcePasses: Pass[] = [
      await planZero(pool, cache, "loopForces", buffers.forces, components * 2),
      await planZero(pool, cache, "loopEnergy", accumulators.energy, 2),
    ];
    for (const [slot, force] of bonded.entries()) {
      forcePasses.push(
        await planBonded(cache, prepareBonded(pool, force, slot), accumulators),
      );
    }
    // Restraints, when the system carries them. The equilibration stage adds
    // them and the production stage does not, so this follows the system rather
    // than a flag.
    const restraints = system.forces.filter(
      (f): f is PositionRestraintForce => f.type === "PositionRestraint",
    );
    const residentRestraints: ResidentRestraint[] = [];
    for (const [slot, force] of restraints.entries()) {
      const resident = prepareRestraint(pool, force, state.boxVectors, slot);
      residentRestraints.push(resident);
      forcePasses.push(await planRestraint(cache, resident, accumulators));
    }
    // Deciding whether the neighbour list is stale, rebuilding it when it is,
    // then the forces over it. `pairs` is the cell-shell path, which with a
    // list handles only the atoms whose list overflowed.
    const { rebuild, listPairs, pairs, exceptions } = await planDirect(
      cache,
      direct,
      accumulators,
    );
    forcePasses.push(...rebuild);
    if (listPairs !== null) forcePasses.push(listPairs);
    forcePasses.push(pairs);
    if (exceptions !== null) forcePasses.push(exceptions);
    forcePasses.push(...(await planPme(pool, cache, pme, accumulators)));

    // The momentum reduction clears its accumulator first, because a
    // fixed-point sum has to start from an exact zero.
    const totals = buffers.totals;
    const reduceBindings = (): readonly { binding: number; resource: GPUBindingResource }[] => {
      const mass = pool.existing("system/mass");
      if (mass === undefined) throw new Error("system/mass is missing from the pool");
      return [
        { binding: 0, resource: { buffer: buffers.velocities } },
        { binding: 1, resource: { buffer: mass } },
        { binding: 2, resource: { buffer: totals } },
        { binding: 3, resource: { buffer: reduceParamsBuffer } },
      ];
    };
    let totalMass = 0;
    for (let i = 0; i < n; i++) totalMass += system.masses[i] as number;
    const reduceBytes = new ArrayBuffer(16);
    new Uint32Array(reduceBytes, 0, 1)[0] = n;
    new Float32Array(reduceBytes, 8, 1)[0] = totalMass === 0 ? 0 : 1 / totalMass;
    const reduceParamsBuffer = pool.buffer({
      label: "loop/reduceParams",
      byteLength: 16,
      usage: uniformUsage(),
    });
    device.queue.writeBuffer(reduceParamsBuffer, 0, new Uint8Array(reduceBytes));

    const reduceWorkgroups: readonly [number] = [Math.ceil(n / 64)];
    const comPasses = [
      await planZero(pool, cache, "loopTotals", totals, 6),
      await cache.pass(
        { label: "reduce/momentum", code: REDUCE, entryPoint: "momentum" },
        reduceBindings(),
        reduceWorkgroups,
      ),
      await cache.pass(
        { label: "reduce/subtract_com", code: REDUCE, entryPoint: "subtract_com" },
        reduceBindings(),
        reduceWorkgroups,
      ),
    ];

    // The ring is allocated once, at its maximum size, because a named buffer's
    // size is fixed for the life of the pool -- and because the phase bind
    // groups reference it, so it cannot be replaced later. `run` chunks a
    // longer request into batches of this size.
    const batch = options.maxBatch ?? DEFAULT_MAX_BATCH;
    const ring = writeStepParamsRing(
      pool,
      system,
      StepLoop.integratorOptions(options, 0),
      batch,
      0,
    );
    const phase = async (entryPoint: string): Promise<Pass> =>
      planPhase(cache, entryPoint, buffers, pool, system, ring);
    const positionConstraints = await planConstraints(
      pool,
      cache,
      tables,
      buffers.positions,
      buffers.trial,
      { tolerance: options.constraintTolerance },
    );
    const velocityConstraints = await planConstraints(
      pool,
      cache,
      tables,
      buffers.positions,
      buffers.velocities,
      { tolerance: options.constraintTolerance, velocities: true },
    );

    let phaseBuilder: (offset: number) => readonly Pass[];
    if (options.integrator === "verlet") {
      const kickDrift = await phase("verlet_kick_drift");
      const finish = await phase("verlet_finish");
      phaseBuilder = (offset) => [
        withOffsets(kickDrift, [offset]),
        ...positionConstraints,
        withOffsets(finish, [offset]),
      ];
    } else {
      const kick = await phase("lfmiddle_kick");
      const drift = await phase("lfmiddle_drift");
      const finish = await phase("lfmiddle_finish");
      phaseBuilder = (offset) => [
        withOffsets(kick, [offset]),
        ...velocityConstraints,
        withOffsets(drift, [offset]),
        ...positionConstraints,
        withOffsets(finish, [offset]),
      ];
    }

    const volume = (box[0] as number) * (box[1] as number) * (box[2] as number);
    const remover = centreOfMassRemover(system);
    // Counted by the reference, which counts them the way OpenMM's
    // StateDataReporter does -- moving particles only, less one per constraint
    // and three for the remover.
    const degreesOfFreedom = countDegreesOfFreedom(system);

    const barostatForce = system.forces.find(
      (f) => f.type === "MonteCarloBarostat",
    );
    const loop = new StepLoop(
      device,
      pool,
      cache,
      system,
      options,
      accumulators,
      buffers,
      {
        dispersion: direct.dispersion,
        self: selfEnergy(nonbonded, alpha),
        netCharge: netChargeCorrection(nonbonded, alpha, volume),
      },
      state.boxVectors,
      alpha,
      nonbonded,
      direct,
      pme,
      residentRestraints,
      system.forces.find(
        (f): f is import("../engine/system.js").MonteCarloBarostat =>
          f.type === "MonteCarloBarostat",
      ),
      degreesOfFreedom,
      remover === undefined ? 0 : remover.frequency,
      wrapPass,
      cellPasses,
      forcePasses,
      comPasses,
      phaseBuilder,
    );
    if (barostatForce !== undefined) {
      loop.barostatState = initialBarostatState(state.boxVectors);
    }
    return loop;
  }

  private static integratorOptions(
    options: RunOptions,
    step: number,
  ): IntegratorOptions {
    return {
      timeStep: options.timeStep,
      constraintTolerance: options.constraintTolerance,
      temperature: options.temperature,
      friction: options.friction,
      ...(options.seed === undefined ? {} : { seed: options.seed }),
      step,
    };
  }

  /** How many steps have run. */
  get steps(): number {
    return this.stepCount;
  }

  /**
   * Continue the step counter from a checkpoint.
   *
   * 🔴 Not cosmetic. Reports land on steps that are multiples of the interval
   * measured on the absolute counter, so a stride that restarted from zero
   * would restart the cadence and its log steps would stop lining up with the
   * previous stride's. The notebook assigns `simulation.currentStep` by hand
   * for exactly this reason. It also seeds the thermostat's generator, so
   * resuming at the right step is what keeps a resumed run from re-drawing the
   * noise it already used.
   */
  advanceStepCount(step: number): void {
    if (!Number.isInteger(step) || step < 0) {
      throw new Error(`a step count must be a non-negative integer, got ${step}`);
    }
    this.stepCount = step;
  }

  /** The barostat's accepted and attempted counts, when there is one. */
  get barostatStatistics(): {
    readonly attempts: number;
    readonly accepted: number;
    readonly volumeScale: number;
  } | null {
    if (this.barostatState === undefined) return null;
    return {
      attempts: this.barostatAttempts,
      accepted: this.barostatAccepted,
      volumeScale: this.barostatState.volumeScale,
    };
  }

  private barostatAttempts = 0;
  private barostatAccepted = 0;

  /**
   * One barostat attempt, if one is due.
   *
   * Driven from the host, which is where OpenMM's API drives it too: it reads
   * the energy, rescales, reads the energy again, and accepts or rejects. Doing
   * the rescaling on the device would save a position round trip, and at a
   * frequency of 25 that trip is about 1 ms against 125 ms of stepping -- so it
   * is not where the time goes, and the two force evaluations are.
   *
   * The deviates come from the same counter-based generator the thermostat
   * uses, on a different domain, so a run is reproducible from its seed. The
   * count is carried rather than derived from the step number because an
   * attempt draws one uniform or two depending on whether the move is uphill.
   */
  private async attemptBarostat(stepsJustRun: number): Promise<void> {
    const barostat = this.barostat;
    const state = this.barostatState;
    if (barostat === undefined || state === undefined) return;
    if (!barostatActsAfterSteps(state, barostat.frequency, stepsJustRun)) return;
    if (this.molecules === undefined) this.molecules = findMolecules(this.system);

    const seed = this.options.seed ?? 0;
    const uniform = (): number => {
      const bits = threefry2x32(
        [this.barostatDraw >>> 0, seed >>> 0],
        [BAROSTAT_DOMAIN, 0],
      );
      this.barostatDraw++;
      return openUnitInterval(bits[0]);
    };

    const positions = await this.positions();
    const attempt = await barostatAttempt(
      this.system,
      barostat,
      state,
      this.boxVectors,
      positions,
      async (trial, trialBox) => {
        this.setBox(trialBox);
        return (await this.potential(trial)).energy;
      },
      uniform,
      this.molecules,
    );

    // `barostatAttempt` leaves the caller to commit, so whichever
    // configuration won is written back -- including on rejection, because the
    // trial was already on the device.
    this.setBox(attempt.boxVectors);
    this.setPositions(attempt.positions);
    this.barostatAttempts++;
    if (attempt.accepted) this.barostatAccepted++;
  }

  /**
   * Encode and submit `steps` steps as one batch.
   *
   * The whole batch is one submission and one validation scope. Awaiting the
   * queue at the end is the only synchronisation: without it the caller cannot
   * know the work finished, and with one per step the loop would spend its time
   * waiting rather than integrating.
   */
  async run(steps: number): Promise<void> {
    if (steps <= 0) return;
    const batch = this.options.maxBatch ?? DEFAULT_MAX_BATCH;
    let remaining = steps;
    while (remaining > 0) {
      const chunk = Math.min(remaining, batch);
      await this.runBatch(chunk);
      remaining -= chunk;
    }
  }

  private async runBatch(steps: number): Promise<void> {
    // 🔴 The barostat cannot be inside an encoded batch. It reads an energy,
    // decides on the host, and may rewrite every box-dependent uniform -- none
    // of which can happen between two dispatches already written into a command
    // buffer. So a batch is cut short at each attempt: the loop submits what it
    // has, runs the attempt, and starts a new batch. At a frequency of 25 that
    // is four submissions per hundred steps instead of one, which is the cost
    // of NPT and is why the notebook's production stage is the slower one.
    if (this.barostat !== undefined && this.barostat.frequency > 0) {
      let done = 0;
      while (done < steps) {
        const until = Math.min(
          steps - done,
          this.barostat.frequency - (this.barostatState?.step ?? 0),
        );
        const chunk = Math.min(Math.max(1, until), steps - done);
        await this.encodeSteps(chunk);
        done += chunk;
        await this.attemptBarostat(chunk);
      }
      return;
    }
    await this.encodeSteps(steps);
  }

  /**
   * GPU time per kernel for one step's worth of passes.
   *
   * 🔴 This **advances the system** by `repeats` steps and leaves the loop's
   * step counter untouched, so the state afterwards does not correspond to any
   * step number. It is a measurement entry point, not part of a run: use it on
   * a loop you are about to discard.
   *
   * `repeats` exists because a single step of a small system is dominated by
   * launch overhead rather than by the kernels. Twenty steps in one submission
   * is enough for the steady-state cost to show through, and the profile
   * reports wall-clock alongside the summed GPU time so the gap between them --
   * which is launch overhead and queue bubbles -- is visible rather than
   * hidden inside a per-kernel number.
   *
   * The cell list is included once rather than per step, matching a run at the
   * default interval of 1; a longer interval would amortise it further.
   */
  async profileStep(repeats = 20): Promise<Profile | undefined> {
    const passes = [
      this.wrapPass,
      ...this.cellPasses,
      ...this.forcePasses,
      ...this.phaseBuilder(0),
    ];
    return profilePasses(this.device, "step", passes, repeats);
  }

  private async encodeSteps(steps: number): Promise<void> {
    // Each block's `step` field feeds the generator, so the ring is rewritten
    // with this batch's step numbers before anything is encoded.
    writeStepParamsRing(
      this.pool,
      this.system,
      StepLoop.integratorOptions(this.options, 0),
      this.options.maxBatch ?? DEFAULT_MAX_BATCH,
      this.stepCount,
    );
    const device = this.device;
    device.pushErrorScope("validation");
    const encoder = device.createCommandEncoder({ label: "step batch" });
    for (let i = 0; i < steps; i++) {
      const step = this.stepCount + i;
      // `updateContextState` runs before the forces, which is where
      // CMMotionRemover acts. See src/reference/com-motion.ts.
      if (this.comFrequency > 0 && removesOnStep(this.comFrequency, step)) {
        encodePasses(encoder, this.comPasses);
      }
      // Both unconditional. The fold is read by the pair kernels every step,
      // and the cell list is read by `direct_pairs` every step -- it handles
      // the atoms whose neighbour list overflowed, at the current positions, so
      // a stale partition would lose exactly the pairs the fallback exists to
      // keep. The *neighbour* list is what a skin makes safe to reuse, and
      // `rebuild` decides that on the device.
      encodePasses(encoder, [this.wrapPass]);
      encodePasses(encoder, this.cellPasses);
      encodePasses(encoder, this.forcePasses);
      encodePasses(encoder, this.phaseBuilder(i * STEP_PARAMS_STRIDE));
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const error = await device.popErrorScope();
    if (error !== null) throw new Error(`step batch: ${error.message}`);
    this.stepCount += steps;
  }

  /**
   * The energies at the current configuration.
   *
   * The potential comes from the accumulator the last step left, plus the three
   * terms the kernels do not compute: the analytic dispersion correction, the
   * Ewald self energy, and the net-charge correction. All three depend on the
   * parameters and the volume and on no coordinate, so they are host constants.
   */
  async report(): Promise<Report> {
    // The accumulator holds the energy of the configuration the *last* step
    // started from, not the one it ended at, so a report that read it directly
    // would pair a potential energy with velocities half a step away from it --
    // which looks exactly like drift. Re-evaluating costs one force
    // evaluation at a reporting interval, which is not where the time goes.
    await this.evaluateForces();

    const energyWords = new Uint32Array(
      await readBuffer(this.device, this.accumulators.energy, 8),
    );
    const potentialEnergy =
      fromFixed(fromWords(energyWords[0] as number, (energyWords[1] as number) | 0)) +
      this.offsets.dispersion +
      this.offsets.self +
      this.offsets.netCharge;

    // The shifted energy needs the forces and a constraint projection, so it is
    // computed on the host by the same reference the kinetic tests cover. At a
    // reporting interval the two readbacks cost nothing, and this keeps the one
    // definition of the convention in one place.
    const components = this.system.particleCount * 3;
    const positions = await this.positions();
    const velocities = await this.velocities();
    const forces = unpackFixed(
      new Uint32Array(await readBuffer(this.device, this.buffers.forces, components * 8)),
      components,
    );
    const unshifted = kineticEnergy(this.system, velocities);
    const shifted = shiftedKineticEnergy(
      this.system,
      positions,
      velocities,
      forces,
      0.5 * this.options.timeStep,
    );
    return {
      step: this.stepCount,
      time: this.stepCount * this.options.timeStep,
      potentialEnergy,
      kineticEnergy: unshifted,
      shiftedKineticEnergy: shifted,
      totalEnergy: potentialEnergy + shifted,
      temperature:
        this.degreesOfFreedom === 0
          ? 0
          : (2 * unshifted) / (this.degreesOfFreedom * 0.008314462618153243),
    };
  }

  /**
   * Replace the coordinates, and clear the integrator's carried residual.
   *
   * The residual belongs to the trajectory that produced the coordinates it was
   * carrying; keeping it across a jump -- a minimiser's line search, a
   * barostat's rejected move -- would apply a correction computed for a
   * configuration that no longer exists.
   */
  setPositions(positions: Float64Array): void {
    const components = this.system.particleCount * 3;
    if (positions.length !== components) {
      throw new Error(
        `expected ${components} coordinates for ${this.system.particleCount} ` +
          `particles, got ${positions.length}`,
      );
    }
    this.device.queue.writeBuffer(
      this.buffers.positions,
      0,
      packPositions(positions) as unknown as BufferSource,
    );
    this.device.queue.writeBuffer(this.buffers.residual, 0, new Float32Array(components));
  }

  /**
   * Replace the velocities.
   *
   * Separate from `setPositions` because the two are set at different points: a
   * run draws velocities once, after minimisation, while positions are replaced
   * on every line-search trial and every rejected barostat move.
   */
  setVelocities(velocities: Float64Array): void {
    const components = this.system.particleCount * 3;
    if (velocities.length !== components) {
      throw new Error(
        `expected ${components} velocity components for ` +
          `${this.system.particleCount} particles, got ${velocities.length}`,
      );
    }
    this.device.queue.writeBuffer(
      this.buffers.velocities,
      0,
      Float32Array.from(velocities) as unknown as BufferSource,
    );
  }

  /**
   * The energy and forces at `positions`, as a minimiser's objective wants them.
   *
   * This is the shape OpenMM's own minimiser uses: it sets the context's
   * positions and asks for `State::Forces | State::Energy` on every evaluation,
   * so driving L-BFGS from the host with the forces on the device is its
   * architecture rather than a concession. The cell list is rebuilt each time,
   * which it has to be -- a line search moves atoms much further than a step
   * does.
   *
   * The energy includes the three terms the kernels leave to the host, so that
   * it is the same number `report()` gives and the same one OpenMM reports.
   */
  async potential(
    positions: Float64Array,
  ): Promise<{ energy: number; forces: Float64Array }> {
    this.setPositions(positions);
    await this.evaluateForces();
    const components = this.system.particleCount * 3;
    const energyWords = new Uint32Array(
      await readBuffer(this.device, this.accumulators.energy, 8),
    );
    const energy =
      fromFixed(fromWords(energyWords[0] as number, (energyWords[1] as number) | 0)) +
      this.offsets.dispersion +
      this.offsets.self +
      this.offsets.netCharge;
    const forces = unpackFixed(
      new Uint32Array(
        await readBuffer(this.device, this.buffers.forces, components * 8),
      ),
      components,
    );
    return { energy, forces };
  }

  /** The cell grid the list was built on, for reports and checks. */
  cellGeometry(): { readonly divisions: readonly number[]; readonly extent: readonly number[] } {
    const { geometry } = this.direct.cellList;
    return { divisions: geometry.divisions, extent: geometry.extent };
  }

  /** The current coordinates, nm. */
  async positions(): Promise<Float64Array> {
    const components = this.system.particleCount * 3;
    const words = new Uint32Array(
      await readBuffer(this.device, this.buffers.positions, components * 8),
    );
    return unpackFixed(words, components);
  }

  /** The current velocities, nm/ps. */
  async velocities(): Promise<Float64Array> {
    const components = this.system.particleCount * 3;
    const values = new Float32Array(
      await readBuffer(this.device, this.buffers.velocities, components * 4),
    );
    return Float64Array.from(values);
  }

  /**
   * Evaluate the forces at the current configuration without stepping.
   *
   * For a report before the first step, and for the energy drift gate, which
   * needs a potential energy that belongs to the coordinates it is about to
   * report rather than to the step before.
   */
  async evaluateForces(): Promise<void> {
    await this.submit("forces", [
      this.wrapPass,
      ...this.cellPasses,
      ...this.forcePasses,
    ]);
  }

  /**
   * The neighbour list's bookkeeping, read back.
   *
   * For reports and measurements, not for the step path: this drains the
   * pipeline, which is exactly what keeping the rebuild decision on the device
   * avoids. `rebuilds` is what makes the skin judgeable -- steps per rebuild is
   * measured rather than assumed -- and `overflowed` says whether any atom fell
   * back to the cell-shell scan.
   */
  async verletFlags(): Promise<{
    readonly rebuilds: number;
    /** The largest displacement since a build that the run ever saw, nm. */
    readonly worstDisplacement: number;
    readonly overflowed: boolean;
    readonly capacity: number;
    readonly skin: number;
    readonly byteLength: number;
  }> {
    const { verlet } = this.direct;
    const words = new Uint32Array(await readBuffer(this.device, verlet.flags, 32));
    return {
      rebuilds: words[3] as number,
      worstDisplacement: (words[4] as number) / 1000,
      overflowed: (words[2] as number) !== 0,
      capacity: verlet.capacity,
      skin: verlet.skin,
      byteLength: verlet.byteLength,
    };
  }

  private async submit(label: string, passes: readonly Pass[]): Promise<void> {
    this.device.pushErrorScope("validation");
    const encoder = this.device.createCommandEncoder({ label });
    encodePasses(encoder, passes);
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const error = await this.device.popErrorScope();
    if (error !== null) throw new Error(`${label}: ${error.message}`);
  }

  /**
   * The PME mesh, which is fixed for the life of the run.
   *
   * Exposed because a comparison against the reference has to use the same
   * mesh: the reference would otherwise choose one from the current box, and
   * after a barostat move that is a different grid and a different energy.
   */
  get mesh(): Vec3 {
    return this.pme.grid;
  }

  /**
   * The current box's diagonal, nm.
   *
   * The diagonal, not the edge lengths: for a skewed cell those differ, and
   * what the DCD header and the protocol want is the former. `boxVectorsNow`
   * is the whole cell.
   */
  get boxLengths(): Vec3 {
    return [
      (this.boxVectors[0] as Vec3)[0] as number,
      (this.boxVectors[1] as Vec3)[1] as number,
      (this.boxVectors[2] as Vec3)[2] as number,
    ];
  }

  /** The whole cell, for anything that cannot assume a rectangular one. */
  get boxVectorsNow(): readonly [Vec3, Vec3, Vec3] {
    return this.boxVectors;
  }

  /**
   * Change the box.
   *
   * Every uniform that carries a box-derived quantity is rewritten: the pair
   * and exception kernels' boxes in both f32 and fixed point, the three
   * reciprocal-space blocks -- whose `prefactor` carries 1/volume -- the cell
   * geometry, and each restraint's wrap. So are the two host-side energy terms
   * that depend on the volume, the dispersion correction and the net-charge
   * correction. The Ewald self energy does not: it is a sum over charges.
   *
   * Two things deliberately do *not* change.
   *
   * The **PME mesh** stays as the starting box chose it, so the grid spacing
   * moves with the volume. That is what OpenMM does -- the mesh is fixed at
   * context creation and `setPeriodicBoxVectors` does not touch it -- and
   * matching it is what keeps an energy comparison meaningful.
   *
   * The **cell divisions** stay too, because the number of cells decides the
   * size of the buffer the pair kernel indexes and that cannot change mid-run.
   * The extents move instead, which is correct while a cell stays at least a
   * cutoff wide; `updateCellBox` throws `BoxTooSmallForCells` rather than
   * quietly returning a list that misses pairs.
   */
  setBox(boxVectors: readonly [Vec3, Vec3, Vec3]): void {
    const box = [
      (boxVectors[0] as Vec3)[0] as number,
      (boxVectors[1] as Vec3)[1] as number,
      (boxVectors[2] as Vec3)[2] as number,
    ] as Vec3;
    const queue = this.device.queue;
    const write = (buffer: GPUBuffer, data: ArrayBufferView): void =>
      queue.writeBuffer(buffer, 0, data as unknown as BufferSource);

    write(
      this.direct.pairParams,
      packPairParams(
        this.nonbonded,
        this.direct.sections,
        this.direct.cellList.geometry.divisions,
        box,
        boxVectors,
        this.alpha,
        this.direct.verlet,
      ),
    );
    write(
      this.direct.exceptionParams,
      packExceptionParams(this.nonbonded, box, boxVectors, this.alpha),
    );
    write(
      this.pme.pmeParams,
      packPmeParams(this.system.particleCount, this.pme.grid, box, this.alpha),
    );
    write(this.pme.convolveParams, packConvolveParams(this.pme.grid, box, this.alpha));
    write(
      this.pme.interpolateParams,
      packInterpolateParams(this.system.particleCount, this.pme.grid, box),
    );
    // The radius `prepareDirect` actually sized the cells by, which is clamped
    // for a small box and so is not always `cutoff + skin`.
    updateCellBox(
      this.pool,
      this.direct.cellList,
      boxVectors,
      this.direct.verlet.searchRadius,
    );
    for (const restraint of this.restraints) {
      write(restraint.params, packRestraintParams(restraint.count, box, boxVectors));
    }

    const volume = (box[0] as number) * (box[1] as number) * (box[2] as number);
    this.offsets = {
      dispersion: this.nonbonded.useDispersionCorrection
        ? dispersionCorrection(this.nonbonded, volume)
        : 0,
      // A sum over charges, so the volume does not enter it.
      self: this.offsets.self,
      netCharge: netChargeCorrection(this.nonbonded, this.alpha, volume),
    };
    this.boxVectors = boxVectors;
  }

  /** The shared force accumulator, for a validator that needs to read it. */
  get forceBuffer(): GPUBuffer {
    return this.buffers.forces;
  }

  /** How many passes a step encodes, for the record. */
  get passesPerStep(): number {
    return this.forcePasses.length + this.phaseBuilder(0).length;
  }

  report_pool(): ReturnType<BufferPool["report"]> {
    return this.pool.report();
  }

  destroy(): { readonly leaked: number } {
    return this.pool.destroy();
  }
}
