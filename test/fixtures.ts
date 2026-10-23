/**
 * Access to the generated reference fixtures.
 *
 * They are not committed: `system.xml` is 7.5 MB for the solvated
 * protein-ligand case, and the preparation that produces them is not
 * reproducible anyway (see docs/NOTEBOOK_PARITY.md finding 2), so a committed
 * copy would be a snapshot pretending to be a specification.  What *is*
 * committed is each run's `facts.json` under docs/measurements/, and the tool
 * that regenerates everything.
 *
 * A test that needs a fixture skips with the command to produce it.
 */

import { readFileSync } from "node:fs";
import { parseState, type OpenMMState } from "../src/engine/state.js";
import { parseSystem, type OpenMMSystem } from "../src/engine/system.js";

export interface Fixture {
  readonly name: string;
  readonly system: OpenMMSystem;
  readonly state: OpenMMState;
  /** What OpenMM reported for this exact system, keyed "<group>:<ForceName>". */
  readonly referenceEnergies: Readonly<Record<string, number>>;
  readonly facts: {
    readonly atoms_total: number;
    readonly constraints: number;
    readonly pme: { readonly alpha_per_nm: number; readonly grid_reference_platform: number[] };
    readonly reference_energies_kJ_mol: Record<string, number>;
    readonly reference_forces_kJ_mol_nm: { readonly rms: number; readonly max_component: number };
  };
}

export const FIXTURE_NAMES = ["1L2Y", "3HTB"] as const;
export type FixtureName = (typeof FIXTURE_NAMES)[number];

const COMMANDS: Record<FixtureName, string> = {
  "1L2Y": "python tools/build_reference_system.py --pdb-id 1L2Y --out fixtures",
  "3HTB":
    "python tools/build_reference_system.py --pdb-id 3HTB --ligand JZ4 --out fixtures",
};

const cache = new Map<string, Fixture | null>();

/** The fixture, or null when it has not been generated. */
export function fixture(name: FixtureName): Fixture | null {
  const cached = cache.get(name);
  if (cached !== undefined) return cached;
  let value: Fixture | null = null;
  try {
    const dir = `fixtures/${name}`;
    const facts = JSON.parse(readFileSync(`${dir}/facts.json`, "utf8")) as Fixture["facts"];
    value = {
      name,
      system: parseSystem(readFileSync(`${dir}/system.xml`, "utf8")),
      state: parseState(readFileSync(`${dir}/state.xml`, "utf8")),
      referenceEnergies: facts.reference_energies_kJ_mol,
      facts,
    };
  } catch {
    process.stderr.write(
      `\n  Fixture ${name} is absent. Generate it with:\n    ${COMMANDS[name]}\n` +
        "  See tools/README.md for the conda environment.\n\n",
    );
  }
  cache.set(name, value);
  return value;
}

/** The reference energy of one force group, by the force's class name. */
export function referenceEnergy(f: Fixture, forceName: string): number {
  const key = Object.keys(f.referenceEnergies).find(
    (k) => k.slice(k.indexOf(":") + 1) === forceName,
  );
  if (key === undefined) {
    throw new Error(
      `${f.name} has no reference energy for ${forceName}; it has ` +
        Object.keys(f.referenceEnergies).join(", "),
    );
  }
  return f.referenceEnergies[key] as number;
}

/**
 * Build something from a fixture, only when the fixture is there.
 *
 * 🔴 **`describe.skipIf(...)` still runs its body.** `skipIf` marks the tests
 * inside as skipped; it does not stop Vitest calling the callback to find them.
 * So a `readFileSync` — or anything parsing one — sitting directly in a skipped
 * `describe` throws at *collection*, which fails the whole file instead of
 * skipping it. And it is invisible on a machine that has the fixtures, because
 * there the read simply works.
 *
 * It was invisible here for exactly that reason. Measured by moving `fixtures/`
 * aside, which is what a fresh clone is: **seven files failed and `npm test`
 * exited 1**, with `ENOENT` and no assertion anywhere. `fixtures/` is
 * gitignored, so that was every clone of this repository — and CI's `check` job
 * and the Pages deploy both run `npm run check`, so neither could ever have
 * gone green.
 *
 * The value is typed as though the fixture were there. Nothing inside a skipped
 * block touches it, so the lie is never observed; the alternative is threading
 * `undefined` through a hundred call sites to work around a quirk of the runner.
 * Pass the *same* condition the `describe` is skipped on — if the two disagree,
 * a running test gets `undefined` and says so immediately.
 */
export function ifPresent<T>(condition: boolean, build: () => T): T {
  return condition ? build() : (undefined as unknown as T);
}
