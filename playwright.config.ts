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
  // 🔴 The long gate runs against the **built** page, not the dev server, and
  // this is not a preference.
  //
  // The dev server watches `src/`, and the run page imports the engine straight
  // out of it. Editing any file in that graph while a run is in flight -- which
  // during a 90-minute gate is likely -- invalidates the module, and with no
  // HMR handlers Vite falls back to a full page reload. That wipes the uploaded
  // files and the form, so the run simply stops, `__mirRun.state` never reaches
  // `done`, and the gate times out with a page snapshot showing default form
  // values and "Nothing is uploaded". It looks like the engine hung. It is the
  // page having been reloaded underneath it.
  //
  // That happened. `vite preview` serves static files with no watcher, so it
  // cannot happen again, and it has the side benefit of testing the artefact
  // users actually get -- which CLAUDE.md notes is a distinct thing, since a
  // page missing from `vite.config.ts` builds in development and vanishes from
  // a production build.
  //
  // `reuseExistingServer` is off for the gate too: attaching to a dev server
  // someone left running would silently undo all of this.
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
