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
  timeout: 180_000,
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
