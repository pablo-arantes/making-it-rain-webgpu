import { defineConfig } from "vitest/config";

/**
 * The slow lane.
 *
 * `*.slow.test.ts` is excluded from the default run and collected here instead.
 * These are not lower-value tests -- the minimiser's parity against OpenMM is
 * one of them -- they are minutes of double-precision PME, because a minimiser
 * calls the force field fifty times and the reference evaluates the mesh in f64
 * on the CPU. A unit suite that takes minutes stops being run, and a test that
 * stops being run is worse than one that is explicitly separate.
 */
export default defineConfig({
  test: {
    include: ["test/**/*.slow.test.ts"],
    exclude: ["test/browser/**", "node_modules/**"],
    environment: "node",
    testTimeout: 600_000,
  },
});
