import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // GPU and oracle lanes are opt-in: they self-skip without their env flag,
    // so `npm test` stays runnable with no GPU and no WebAssembly OpenMM.
    //
    // `*.slow.test.ts` is excluded and has its own `npm run test:slow`. It is
    // not a second-class lane -- the minimiser's parity test lives there -- it
    // is a minute of double-precision PME per run, and a unit suite that takes
    // a minute stops being run.
    exclude: ["test/browser/**", "test/**/*.slow.test.ts", "node_modules/**"],
    environment: "node",
    testTimeout: 60_000,
  },
});
