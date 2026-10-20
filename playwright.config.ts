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
  // The long gate runs against the **built** page rather than the dev server.
  //
  // Two reasons, and the weaker one is listed second on purpose. The dev server
  // watches `src/`, and the run page imports the engine straight out of it
  // (`web/run.ts` -> `src/engine/protocol.ts` -> `src/analysis/pdb.ts`). Editing
  // a file in that graph invalidates the module, and with no HMR handlers Vite
  // falls back to a full page reload -- which during a 90-minute run would drop
  // the uploaded files and stop it dead. A gate that long should not share a
  // file watcher with someone's editor. Second, it exercises the artefact users
  // actually get, which CLAUDE.md notes is a distinct thing: a page missing from
  // `vite.config.ts` builds in development and vanishes from a production build.
  //
  // 🔴 This is *not* a diagnosed fix for a specific failure. One 90-minute gate
  // did time out while `src/analysis/pdb.ts` was edited mid-run, and the reload
  // is a plausible cause -- but it was never established. The evidence first
  // cited for it, a snapshot reading "Nothing is uploaded", turned out to be a
  // status line that was set once at start-up and never updated; a healthy
  // running run shows the identical string. The page has since been fixed, and
  // the cause of that timeout remains unknown.
  //
  // `reuseExistingServer` is off for the gate: attaching to a dev server
  // someone left running would silently undo the first reason.
  webServer:
    process.env["MIR_LONG_RUN"] === "1"
      ? {
          command: "npm run build:web && npm run preview",
          url: "http://localhost:4173",
          reuseExistingServer: false,
          timeout: 300_000,
        }
      : {
          command: "npm run dev",
          url: "http://localhost:4173",
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
});
