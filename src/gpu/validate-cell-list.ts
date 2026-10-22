/**
 * The device-built cell list against the host-built one.
 *
 * Three claims, and the third is the one that matters.
 *
 * **The partition is the same.** Every atom lands in the same cell the host
 * puts it in, and each cell holds the same number. That is not quite
 * guaranteed: the device buckets by comparing against accumulated multiples of
 * the cell extent in fixed point, the host divides in f64, and the two can
 * disagree for an atom within about 2e-9 nm of a boundary. So the count of
 * disagreements is reported rather than asserted to be zero, and a handful
 * would be acceptable while a systematic difference would not.
 *
 * **The list is well formed.** Every atom appears exactly once, each atom sits
 * inside the span its own cell owns, and the final offset is the atom count.
 * A counting sort that drops or duplicates an atom is otherwise invisible: the
 * forces just come out slightly wrong.
 *
 * **The forces are bit-identical.** Not because the order cannot matter -- it
 * does, which is the finding this test produced. The first version of the cell
 * list left the atomic scatter's order alone, on the argument that the forces
 * accumulate in exact fixed point. They do, but `nonbonded-pairs.wgsl` sums one
 * atom's neighbours in an **f32 register** first and makes a single fixed-point
 * add at the end, so the per-atom sum reassociates with the neighbour order.
 * This test measured it at 1.89e-3 kJ/mol/nm, about 2e-6 of the force RMS: too
 * small to see in a force comparison, large enough to make two runs of the same
 * input differ. `sort_cells` now puts each cell in ascending atom index, which
 * is also what the host builder produces -- so the two lists are the same
 * permutation and the forces agree to the last bit. The assertion is `=== 0`
 * rather than a tolerance, because anything else means the order drifted again.
 */

import { parseState } from "../engine/state.js";
import { parseSystem, type OpenMMSystem, type NonbondedForce } from "../engine/system.js";
import { ewaldAlpha } from "../engine/units.js";
import { rectangularBox } from "../reference/forces/nonbonded.js";
import { BufferPool } from "./allocator.js";
import { buildCellListOnDevice, readCellList } from "./cell-list.js";
import {
  buildCellList,
  evaluateDirectSpace,
  prepareDirect,
  type CellList,
} from "./nonbonded.js";
import { storageUsage } from "./usage.js";
import { packPositions } from "./forces.js";

export interface CellListComparison {
  readonly particleCount: number;
  readonly divisions: readonly number[];
  readonly cellCount: number;
  /** Atoms the device put in a different cell than the host did. */
  readonly assignmentDifferences: number;
  /** Cells whose population differs. */
  readonly populationDifferences: number;
  /** True when every atom appears exactly once in the sorted order. */
  readonly permutationValid: boolean;
  /** True when every atom sits within the span its cell owns. */
  readonly spansValid: boolean;
  readonly totalMatches: boolean;
  /** How many atoms the device ordered differently inside their cell. */
  readonly orderDifferences: number;
  /** kJ/mol, with the host list and with the device list. */
  readonly hostEnergy: number;
  readonly deviceEnergy: number;
  /** The worst force component difference between the two. Must be exactly 0. */
  readonly forceDifference: number;
  readonly ok: boolean;
}

export async function validateCellList(
  device: GPUDevice,
  texts: { readonly systemXml: string; readonly stateXml: string },
): Promise<CellListComparison> {
  const system: OpenMMSystem = parseSystem(texts.systemXml);
  const state = parseState(texts.stateXml);
  const force = system.forces.find(
    (f): f is NonbondedForce => f.type === "NonbondedForce",
  );
  if (force === undefined) throw new Error("the fixture has no NonbondedForce");
  const box = rectangularBox(state.boxVectors);
  const particleCount = system.particleCount;
  const positionWords = packPositions(state.positions);

  const pool = new BufferPool(device);
  try {
    // Built through the same `prepareDirect` the forces use, so the layout under
    // test is the layout a run uses rather than a parallel one.
    const alpha = ewaldAlpha(force.cutoff, force.ewaldTolerance);
    // No list: this check is about the cell list, and the shell path is what
    // reads it.
    const resident = prepareDirect(pool, force, state.boxVectors, alpha, 0);
    const positions = pool.buffer({
      label: "cell/positions",
      byteLength: positionWords.byteLength,
      usage: storageUsage(),
    });
    device.queue.writeBuffer(
      positions,
      0,
      positionWords as unknown as BufferSource,
    );
    await buildCellListOnDevice(pool, resident.cellList, positions);
    const built = await readCellList(pool, resident.cellList);
    const host = buildCellList(state.positions, box, force.cutoff);

    let assignmentDifferences = 0;
    for (let atom = 0; atom < particleCount; atom++) {
      // The host's cell index is recoverable from its own sorted order.
      if ((built.cellOf[atom] as number) !== hostCellOf(host, atom)) {
        assignmentDifferences++;
      }
    }
    let populationDifferences = 0;
    for (let cell = 0; cell <= host.cellCount; cell++) {
      if ((built.starts[cell] as number) !== (host.starts[cell] as number)) {
        populationDifferences++;
      }
    }

    const seen = new Uint8Array(particleCount);
    let permutationValid = true;
    for (const atom of built.order) {
      if (atom >= particleCount || seen[atom] === 1) permutationValid = false;
      else seen[atom] = 1;
    }
    if (seen.some((v) => v === 0)) permutationValid = false;

    let spansValid = true;
    for (let cell = 0; cell < host.cellCount; cell++) {
      const begin = built.starts[cell] as number;
      const end = built.starts[cell + 1] as number;
      for (let slot = begin; slot < end; slot++) {
        if ((built.cellOf[built.order[slot] as number] as number) !== cell) {
          spansValid = false;
        }
      }
    }
    const totalMatches = (built.starts[host.cellCount] as number) === particleCount;

    let orderDifferences = 0;
    for (let slot = 0; slot < particleCount; slot++) {
      if ((built.order[slot] as number) !== (host.order[slot] as number)) {
        orderDifferences++;
      }
    }

    const deviceList: CellList = {
      cells: resident.cellList.geometry.divisions,
      cellCount: resident.cellList.geometry.cellCount,
      starts: built.starts,
      order: built.order,
    };
    const withHost = await evaluateDirectSpace(
      device,
      force,
      positionWords,
      state.positions,
      state.boxVectors,
      alpha,
      host,
    );
    const withDevice = await evaluateDirectSpace(
      device,
      force,
      positionWords,
      state.positions,
      state.boxVectors,
      alpha,
      deviceList,
    );
    let forceDifference = 0;
    for (let i = 0; i < withHost.forces.length; i++) {
      forceDifference = Math.max(
        forceDifference,
        Math.abs((withHost.forces[i] as number) - (withDevice.forces[i] as number)),
      );
    }

    return {
      particleCount,
      divisions: resident.cellList.geometry.divisions,
      cellCount: resident.cellList.geometry.cellCount,
      assignmentDifferences,
      populationDifferences,
      permutationValid,
      spansValid,
      totalMatches,
      orderDifferences,
      hostEnergy: withHost.energy,
      deviceEnergy: withDevice.energy,
      forceDifference,
      ok:
        assignmentDifferences === 0 &&
        populationDifferences === 0 &&
        permutationValid &&
        spansValid &&
        totalMatches &&
        // Bit-identical, not "close": an exact integer sum has no excuse.
        withHost.energy === withDevice.energy &&
        forceDifference === 0,
    };
  } finally {
    pool.destroy();
  }
}

/** Which cell the host builder put an atom in, read back from its order. */
function hostCellOf(host: CellList, atom: number): number {
  const cached = hostCellCache.get(host);
  if (cached !== undefined) return cached[atom] as number;
  const table = new Uint32Array(host.order.length);
  for (let cell = 0; cell < host.cellCount; cell++) {
    for (
      let slot = host.starts[cell] as number;
      slot < (host.starts[cell + 1] as number);
      slot++
    ) {
      table[host.order[slot] as number] = cell;
    }
  }
  hostCellCache.set(host, table);
  return table[atom] as number;
}

const hostCellCache = new WeakMap<CellList, Uint32Array>();
