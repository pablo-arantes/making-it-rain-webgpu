/**
 * Host side of the constraint kernels.
 *
 * Packs the cluster tables once -- they depend on the system, never on the
 * coordinates -- and runs SETTLE then SHAKE, exactly as the reference routes
 * them.  Buffers come from the pool, so a step loop allocates nothing per step.
 */

import { dispatch } from "./compute.js";
import { SETTLE, SHAKE } from "./kernels.js";
import { BufferPool } from "./allocator.js";
import { MAX_CONSTRAINT_ITERATIONS } from "../reference/constraints.js";
import type { ConstraintSplit } from "../engine/constraint-clusters.js";
import type { OpenMMSystem } from "../engine/system.js";

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

/** Flat tables for the kernels, derived once from the system. */
export interface ConstraintTables {
  readonly settleCount: number;
  readonly settleAtoms: Uint32Array;
  readonly settleDistances: Float32Array;
  readonly clusterCount: number;
  readonly headers: Uint32Array;
  readonly atoms: Uint32Array;
  readonly constraints: Float32Array;
  readonly mass: Float32Array;
  readonly inverseMass: Float32Array;
}

export function packConstraints(
  system: OpenMMSystem,
  split: ConstraintSplit,
): ConstraintTables {
  const settleAtoms = new Uint32Array(Math.max(1, split.settle.length) * 4);
  const settleDistances = new Float32Array(Math.max(1, split.settle.length) * 2);
  split.settle.forEach((cluster, i) => {
    settleAtoms[i * 4] = cluster.central;
    settleAtoms[i * 4 + 1] = cluster.first;
    settleAtoms[i * 4 + 2] = cluster.second;
    settleDistances[i * 2] = cluster.distance1;
    settleDistances[i * 2 + 1] = cluster.distance2;
  });

  const clusters = split.iterativeClusters;
  const headers = new Uint32Array(Math.max(1, clusters.length) * 4);
  const atomList: number[] = [];
  // Four floats a constraint: the two *local* atom indices as raw bits, then
  // the target distance.
  const constraintList: number[] = [];
  clusters.forEach((cluster, i) => {
    const atomOffset = atomList.length;
    const constraintOffset = constraintList.length / 4;
    headers[i * 4] = cluster.atoms.length;
    headers[i * 4 + 1] = cluster.constraints.length;
    headers[i * 4 + 2] = atomOffset;
    headers[i * 4 + 3] = constraintOffset;
    atomList.push(...cluster.atoms);
    for (const c of cluster.constraints) {
      const p1 = system.constraintP1[c] as number;
      const p2 = system.constraintP2[c] as number;
      constraintList.push(
        cluster.atoms.indexOf(p1),
        cluster.atoms.indexOf(p2),
        system.constraintDistance[c] as number,
        0,
      );
    }
  });
  const constraints = new Float32Array(Math.max(4, constraintList.length));
  const bits = new Uint32Array(constraints.buffer);
  constraintList.forEach((value, k) => {
    // Columns 0 and 1 are indices and travel as bits; 2 and 3 are floats.
    if (k % 4 < 2) bits[k] = value;
    else constraints[k] = value;
  });

  const mass = Float32Array.from(system.masses);
  const inverseMass = Float32Array.from(system.masses, (m) => (m === 0 ? 0 : 1 / m));
  return {
    settleCount: split.settle.length,
    settleAtoms,
    settleDistances,
    clusterCount: clusters.length,
    headers,
    atoms: Uint32Array.from(atomList.length === 0 ? [0] : atomList),
    constraints,
    mass,
    inverseMass,
  };
}

/** Upload the tables into pool-owned buffers. Idempotent by label. */
export function uploadConstraintTables(
  pool: BufferPool,
  tables: ConstraintTables,
): void {
  const put = (label: string, data: ArrayBufferView): void => {
    const buffer = pool.buffer({ label, byteLength: data.byteLength, usage: STORAGE });
    pool.device.queue.writeBuffer(buffer, 0, data as unknown as BufferSource);
  };
  put("constraint/settleAtoms", tables.settleAtoms);
  put("constraint/settleDistances", tables.settleDistances);
  put("constraint/headers", tables.headers);
  put("constraint/atoms", tables.atoms);
  put("constraint/constraints", tables.constraints);
  // Neutral labels: the integrator and the reductions need the same two
  // tables, and the pool is idempotent by label, so they upload once.
  put("system/mass", tables.mass);
  put("system/inverseMass", tables.inverseMass);
}

export interface ConstrainOptions {
  readonly tolerance: number;
  readonly maxIterations?: number;
  /**
   * Project velocities rather than positions.
   *
   * Then `trial` is the f32 velocity buffer and `reference` the fixed-point
   * positions they belong to -- two different formats, which is why the kernel
   * has a separate entry point and a separate binding for each.
   */
  readonly velocities?: boolean;
}

/**
 * Project `trial` onto the constraint surface on the device.
 *
 * `reference` and `trial` are pool-owned fixed-point position buffers. In
 * velocity mode `trial` is instead the f32 velocity buffer, and `reference`
 * stays the positions those velocities belong to -- the reference bond vectors
 * come from there.
 */
export async function constrainOnDevice(
  pool: BufferPool,
  tables: ConstraintTables,
  reference: GPUBuffer,
  trial: GPUBuffer,
  options: ConstrainOptions,
): Promise<void> {
  const device = pool.device;
  const velocityMode = options.velocities === true;
  const need = (label: string): GPUBuffer => {
    const buffer = pool.existing(label);
    if (buffer === undefined) {
      throw new Error(`${label} is not in the pool; call uploadConstraintTables first`);
    }
    return buffer;
  };

  // SETTLE handles the rigid triangles in both modes, analytically and with no
  // tolerance, exactly as the reference routes them. It has to: the iterative
  // kernel's tables hold only the *iterative* clusters, so leaving the
  // triangles to it means leaving them unconstrained -- which cost 2.36 nm/ps
  // on a thermostatted step before the staged diagnostic found it.
  if (tables.settleCount > 0) {
    const params = pool.buffer({
      label: "constraint/settleParams",
      byteLength: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(params, 0, new Uint32Array([tables.settleCount]));
    // The two entry points read the state they project from different
    // bindings -- fixed-point positions at 1, f32 velocities at 6 -- so each
    // is given only what it reaches. `layout: "auto"` derives a layout per
    // entry point and rejects a bind group carrying anything more.
    const settleCommon = [
      { binding: 0, resource: { buffer: reference } },
      { binding: 2, resource: { buffer: need("system/mass") } },
      { binding: 3, resource: { buffer: need("constraint/settleAtoms") } },
      { binding: 5, resource: { buffer: params } },
    ];
    await dispatch(device, {
      label: velocityMode ? "settle/velocities" : "settle/positions",
      code: SETTLE,
      entryPoint: velocityMode ? "settle_velocities" : "settle_cluster",
      // The velocity solve needs no target distances: the bond directions come
      // from the reference positions, and the distances are already implicit
      // in them. Passing a binding an entry point does not reach is an error.
      bindings: velocityMode
        ? [...settleCommon, { binding: 6, resource: { buffer: trial } }]
        : [
            ...settleCommon,
            { binding: 1, resource: { buffer: trial } },
            { binding: 4, resource: { buffer: need("constraint/settleDistances") } },
          ],
      workgroups: [Math.ceil(tables.settleCount / 64)],
    });
  }

  if (tables.clusterCount === 0) return;
  const shakeParams = new ArrayBuffer(16);
  new Uint32Array(shakeParams, 0, 2).set([
    tables.clusterCount,
    options.maxIterations ?? MAX_CONSTRAINT_ITERATIONS,
  ]);
  new Float32Array(shakeParams, 8, 1)[0] = options.tolerance;
  const params = pool.buffer({
    label: "constraint/shakeParams",
    byteLength: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, new Uint8Array(shakeParams));
  // The two entry points take their state from different bindings, so each
  // gets only the ones it reaches: `layout: "auto"` derives a layout per entry
  // point and rejects a bind group carrying anything else.
  const common = [
    { binding: 0, resource: { buffer: reference } },
    { binding: 2, resource: { buffer: need("system/inverseMass") } },
    { binding: 3, resource: { buffer: need("constraint/headers") } },
    { binding: 4, resource: { buffer: need("constraint/atoms") } },
    { binding: 5, resource: { buffer: need("constraint/constraints") } },
    { binding: 6, resource: { buffer: params } },
  ];
  await dispatch(device, {
    label: velocityMode ? "rattle" : "shake",
    code: SHAKE,
    entryPoint: velocityMode ? "rattle_cluster" : "shake_cluster",
    bindings: velocityMode
      ? [...common, { binding: 7, resource: { buffer: trial } }]
      : [...common, { binding: 1, resource: { buffer: trial } }],
    workgroups: [Math.ceil(tables.clusterCount / 64)],
  });
}
