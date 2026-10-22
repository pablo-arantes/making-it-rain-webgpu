/**
 * The step loop as a `ProtocolEngine`.
 *
 * A thin adapter, and thin on purpose: everything it exposes already exists on
 * `StepLoop`, and what it adds is the shape the protocol asks for. Keeping the
 * protocol behind an interface is what lets its orchestration be tested in Node
 * against a stub -- interval arithmetic and resume do not need a GPU -- while
 * the same code drives the real thing in a browser.
 */

import type { ProtocolEngine } from "../engine/protocol.js";
import type { OpenMMSystem, Vec3 } from "../engine/system.js";
import { minimiseOnDevice } from "./minimise.js";
import type { StepLoop } from "./loop.js";

export class LoopEngine implements ProtocolEngine {
  constructor(
    private readonly loop: StepLoop,
    private readonly system: OpenMMSystem,
  ) {}

  get particleCount(): number {
    return this.system.particleCount;
  }

  get steps(): number {
    return this.loop.steps;
  }

  positions(): Promise<Float64Array> {
    return this.loop.positions();
  }

  velocities(): Promise<Float64Array> {
    return this.loop.velocities();
  }

  boxLengths(): Vec3 {
    return this.loop.boxLengths;
  }

  boxVectors(): readonly [Vec3, Vec3, Vec3] {
    return this.loop.boxVectorsNow;
  }

  setPositions(positions: Float64Array): void {
    this.loop.setPositions(positions);
  }

  setVelocities(velocities: Float64Array): void {
    this.loop.setVelocities(velocities);
  }

  /**
   * Rescale the cell.
   *
   * Takes the diagonal, which is what the protocol has, and keeps the cell's
   * shape: a skewed cell is scaled component-wise so its off-diagonals move
   * with it. Passing three edge lengths and reconstructing a rectangular cell
   * -- which is what this did -- discarded the shape on every call.
   */
  setBox(box: Vec3): void {
    const current = this.loop.boxVectorsNow;
    const scale = [0, 1, 2].map(
      (axis) => (box[axis] as number) / ((current[axis] as Vec3)[axis] as number),
    );
    this.loop.setBox(
      current.map((row, axis) =>
        (row as Vec3).map((value) => value * (scale[axis] as number)),
      ) as unknown as readonly [Vec3, Vec3, Vec3],
    );
  }

  run(steps: number): Promise<void> {
    return this.loop.run(steps);
  }

  async sample(): Promise<{
    potentialEnergy: number;
    kineticEnergy: number;
    totalEnergy: number;
    temperature: number;
  }> {
    const report = await this.loop.report();
    return {
      potentialEnergy: report.potentialEnergy,
      // The unshifted energy, which is what `LangevinIntegrator` reports and so
      // what the notebook's logs contain. `report.totalEnergy` uses the shifted
      // one because that is the conserved quantity; a log is not a conservation
      // check. See docs/DYNAMICS.md.
      kineticEnergy: report.kineticEnergy,
      totalEnergy: report.potentialEnergy + report.kineticEnergy,
      temperature: report.temperature,
    };
  }

  async minimise(maxIterations: number, tolerance: number): Promise<number> {
    const result = await minimiseOnDevice(this.loop, this.system, {
      maxIterations,
      tolerance,
    });
    return result.energy;
  }
}
