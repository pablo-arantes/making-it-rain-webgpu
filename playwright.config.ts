import { defineConfig } from "@playwright/test";

/**
 * The browser lane, and on macOS the only GPU lane there is: no published Dawn
 * build loads on macOS 14 (see CLAUDE.md), so these tests are what actually
 * runs the kernels here.
 *
 * `channel: "chrome"` rather than Playwright's bundled Chromium, because the
 * bundled build does not expose WebGPU. WebGPU also requires a secure context,
 * which is why the tests load over http://localhost rather than about:blank.
 */
export default defineConfig({
  testDir: "test/browser",
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? [["github"], ["list"]] : [["list"]],
  // The page runs the whole engine on both fixtures: every force against
  // OpenMM, the constraints, the cell list, both integrators over eight
  // captured frames, a 500-step conservation run, and five minimisations up to
  // 200 iterations. That is minutes, not seconds, and a timeout set for a
  // single-point energy check turns a slow pass into a confusing failure.
  timeout: 900_000,
  use: {
    baseURL: "http://localhost:4173",
    channel: "chrome",
    launchOptions: { args: ["--enable-unsafe-webgpu"] },
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
