/**
 * Will this system run on this GPU? Checked against devices we do not have.
 *
 * 🔴 Everything in this project is measured on one machine with one GPU, and
 * `CLAUDE.md` says not to act on the Apple-specific findings until they are
 * measured elsewhere. This is the part of cross-device behaviour that *can* be
 * tested from here: the fitness arithmetic is pure, so it runs against the
 * limits other classes of device actually report rather than against the one in
 * front of us.
 *
 * The profile that matters most is `floor` — WebGPU's mandated minimum, which a
 * conformant implementation may expose and none may go below. An engine that
 * silently needs more than the floor works on the author's laptop and fails on
 * a user's, which is the failure mode this exists to prevent.
 *
 * What it does **not** do is tell us the kernels produce the right numbers on
 * another GPU. Only running them there does that, and `npm run test:browser`
 * on such a machine is the lane for it.
 */

import { describe, expect, it } from "vitest";
import {
  DEVICE_PROFILES,
  deviceFitness,
  estimateMaxAtoms,
  KERNEL_WORKGROUP_SIZE,
  MAX_STORAGE_BUFFERS_USED,
} from "../src/gpu/device.js";

/** The sizes this project actually runs, for reference. */
const SYSTEMS = {
  "1L2Y solvated": 2373,
  "3HTB solvated": 26749,
  "3HTB with a 10 Å box": 34884,
} as const;

describe("device fitness", () => {
  it("runs every system this project has on WebGPU's mandated floor", () => {
    // The floor is the guarantee. If a real fixture does not fit it, the engine
    // is not portable and the right answer is to say so here rather than to
    // discover it on someone else's laptop.
    for (const [name, atoms] of Object.entries(SYSTEMS)) {
      const fitness = deviceFitness(DEVICE_PROFILES.floor, atoms);
      expect(
        fitness.fits,
        `${name} (${atoms} atoms) does not fit the floor: ` +
          fitness.problems.map((p) => p.detail).join("; "),
      ).toBe(true);
    }
  });

  it("names the limit that binds, rather than failing generically", () => {
    // Deliberately past the floor's storage binding, which is what the
    // neighbour list reaches first.
    const floorMax = estimateMaxAtoms(
      DEVICE_PROFILES.floor as unknown as GPUSupportedLimits,
    );
    const tooBig = deviceFitness(DEVICE_PROFILES.floor, floorMax.maxAtoms + 1000);
    expect(tooBig.fits).toBe(false);
    expect(tooBig.problems.length).toBeGreaterThan(0);
    // The message has to carry the numbers, or a user cannot act on it.
    const problem = tooBig.problems[0]!;
    expect(problem.needs).toBeGreaterThan(problem.has);
    expect(problem.detail).toMatch(/MB/);
    expect(problem.limit).toMatch(/^max/);
  });

  it("holds more atoms on bigger devices, in the order you would expect", () => {
    const atoms = (profile: Readonly<Record<string, number>>): number =>
      deviceFitness(profile, 1).maxAtoms;
    expect(atoms(DEVICE_PROFILES.floor)).toBeLessThan(
      atoms(DEVICE_PROFILES.integrated),
    );
    expect(atoms(DEVICE_PROFILES.integrated)).toBeLessThanOrEqual(
      atoms(DEVICE_PROFILES.discrete),
    );
    // Even the floor holds a system far larger than the notebook's defaults.
    expect(atoms(DEVICE_PROFILES.floor)).toBeGreaterThan(100_000);
  });

  it("checks the dispatch width, which memory hides", () => {
    // 🔴 Every kernel declares @workgroup_size(64) and dispatches one
    // workgroup per 64 atoms, so the ceiling is
    // maxComputeWorkgroupsPerDimension * 64 -- 4,194,240 atoms at the floor.
    // Memory binds long before that on every real profile, which is exactly
    // why this is easy to forget: it is unreachable today and would become a
    // silent cap if a future device had far more memory than dispatch width.
    const dispatchCeiling =
      DEVICE_PROFILES.floor.maxComputeWorkgroupsPerDimension * KERNEL_WORKGROUP_SIZE;
    expect(dispatchCeiling).toBe(4_194_240);
    expect(deviceFitness(DEVICE_PROFILES.floor, 1).maxAtoms).toBeLessThan(
      dispatchCeiling,
    );

    // A hypothetical device with plenty of memory and a narrow dispatch: the
    // problem must be named, not masked by the memory check passing.
    const narrow = {
      ...DEVICE_PROFILES.discrete,
      maxComputeWorkgroupsPerDimension: 1024,
    };
    const fitness = deviceFitness(narrow, 200_000);
    expect(fitness.fits).toBe(false);
    expect(fitness.problems.map((p) => p.limit)).toContain(
      "maxComputeWorkgroupsPerDimension",
    );
    expect(fitness.dispatchWorkgroups).toBe(Math.ceil(200_000 / 64));
  });

  it("refuses a device below WebGPU's own minimums", () => {
    // Not hypothetical defensiveness: a non-conformant or emulated device can
    // report these, and the failure would otherwise be a bind-group validation
    // error naming a buffer rather than the device.
    const starved = {
      ...DEVICE_PROFILES.floor,
      maxStorageBuffersPerShaderStage: 4,
      maxComputeInvocationsPerWorkgroup: 32,
    };
    const fitness = deviceFitness(starved, 2373);
    expect(fitness.fits).toBe(false);
    const limits = fitness.problems.map((p) => p.limit);
    expect(limits).toContain("maxStorageBuffersPerShaderStage");
    expect(limits).toContain("maxComputeInvocationsPerWorkgroup");
    expect(MAX_STORAGE_BUFFERS_USED).toBe(8);
  });

  it("agrees with the limits the real device reported", () => {
    // The M3 Pro this was developed on, from
    // docs/measurements/device-M3Pro-chrome.json. A regression in the
    // arithmetic shows up as this machine suddenly not fitting its own
    // fixtures.
    const m3Pro = {
      maxBufferSize: 4294967296,
      maxStorageBufferBindingSize: 4294967292,
      maxComputeWorkgroupStorageSize: 32768,
      maxComputeInvocationsPerWorkgroup: 1024,
      maxComputeWorkgroupSizeX: 1024,
      maxComputeWorkgroupsPerDimension: 65535,
      maxStorageBuffersPerShaderStage: 10,
    };
    const fitness = deviceFitness(m3Pro, 26749);
    expect(fitness.fits).toBe(true);
    expect(fitness.problems).toEqual([]);
    // Its dispatch width caps it well below its memory, which is the one case
    // where the dimension limit is the nearer of the two.
    expect(65535 * KERNEL_WORKGROUP_SIZE).toBe(4_194_240);
  });
});
