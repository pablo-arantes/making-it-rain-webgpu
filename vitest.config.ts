import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    // GPU and oracle lanes are opt-in: they self-skip without their env flag,
    // so `npm test` stays runnable with no GPU and no WebAssembly OpenMM.
    exclude: ["test/browser/**", "node_modules/**"],
    environment: "node",
    testTimeout: 60_000,
  },
});
