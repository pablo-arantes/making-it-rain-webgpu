/**
 * Serialising a `State`, and reading it back.
 *
 * This is the protocol's checkpoint: written after every stage and stride, and
 * read back to resume -- which is the notebook's own mechanism, where a stride
 * whose `.rst` already exists is skipped and a disconnected Colab session picks
 * up where it stopped.
 *
 * Two properties, and the second is the one that needs OpenMM. The file must
 * round-trip through *our* parser exactly, and it must be readable by OpenMM so
 * that a run started in a browser can be continued in Python.
 * `tools/check_state.py` does the second; this writes the file it reads.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseState, serialiseState, type OpenMMState } from "../src/engine/state.js";
import type { Vec3 } from "../src/engine/system.js";
import { fixture } from "./fixtures.js";

const boxVectors: readonly [Vec3, Vec3, Vec3] = [
  [3.080639849905961, 0, 0],
  [0, 3.080639849905961, 0],
  [0, 0, 3.080639849905961],
];

/** Values chosen to be awkward: a negative, a denormal-ish, and a long decimal. */
const made: OpenMMState = {
  openmmVersion: "8.4",
  stepCount: 12500,
  time: 25.000000000000004,
  boxVectors,
  particleCount: 3,
  positions: Float64Array.from([
    -1.2345678901234567, 0, 3.141592653589793, 1e-12, 2.5, -0.000123456789,
    6.02e-7, 1.9999999999999998, 0,
  ]),
  velocities: Float64Array.from([
    -0.5000000000000004, 0.5000000000000004, 0, 0.24999999999997247,
    -0.24999999999997247, 0.12500000000004174, 1.5, -1.5, 0.75,
  ]),
};

describe("serialiseState", () => {
  it("round-trips every value exactly through our own parser", () => {
    const parsed = parseState(serialiseState(made));
    expect(parsed.stepCount).toBe(made.stepCount);
    expect(parsed.time).toBe(made.time);
    expect(parsed.particleCount).toBe(made.particleCount);
    expect(Array.from(parsed.positions)).toEqual(Array.from(made.positions));
    expect(Array.from(parsed.velocities!)).toEqual(Array.from(made.velocities!));
    expect(parsed.boxVectors).toEqual(made.boxVectors);
  });

  it("omits velocities when the state has none", () => {
    const { velocities: _omitted, ...withoutVelocities } = made;
    const xml = serialiseState(withoutVelocities as OpenMMState);
    expect(xml).not.toContain("<Velocities>");
    // 🔴 And the parser must report their absence rather than inventing zeros:
    // a state with no velocities means "set them yourself", and zeros would
    // start a run at absolute zero without saying so.
    expect(parseState(xml).velocities).toBeUndefined();
  });

  it("omits forces unless asked, and writes them when asked", () => {
    const withForces = { ...made, forces: Float64Array.from([1, 2, 3, 4, 5, 6, 7, 8, 9]) };
    expect(serialiseState(withForces)).not.toContain("<Forces>");
    const xml = serialiseState(withForces, { includeForces: true });
    expect(xml).toContain("<Forces>");
    expect(Array.from(parseState(xml).forces!)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("round-trips a real fixture", () => {
    const f = fixture("1L2Y");
    if (f === null) return;
    const parsed = parseState(serialiseState(f.state, { includeForces: true }));
    expect(Array.from(parsed.positions)).toEqual(Array.from(f.state.positions));
    expect(parsed.boxVectors).toEqual(f.state.boxVectors);
  });

  it("writes a file for tools/check_state.py to hand to OpenMM", () => {
    // A test with a side effect, deliberately: the Python check needs a file
    // this writer produced, and generating it from the test keeps the two from
    // drifting apart.
    const dir = "fixtures/state-roundtrip";
    mkdirSync(dir, { recursive: true });
    writeFileSync(`${dir}/ours.xml`, serialiseState(made));
    writeFileSync(
      `${dir}/expected.json`,
      JSON.stringify(
        {
          stepCount: made.stepCount,
          time: made.time,
          positions: Array.from(made.positions),
          velocities: Array.from(made.velocities!),
          box: made.boxVectors.flat(),
        },
        null,
        1,
      ),
    );
    expect(true).toBe(true);
  });
});
