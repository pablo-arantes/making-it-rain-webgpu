/**
 * Run the application in a real browser and read a result off the page.
 *
 * This exists because the Node GPU lane does not work here: every published
 * `webgpu` (Dawn) binary from 0.3.10 to 0.6.0 is built against macOS 15 or 26
 * and fails to load on macOS 14 with a missing `MTLLogStateDescriptor`.  Chrome
 * is the lane that works, so the tools use it.
 *
 * Two details that cost time if missed.  `navigator.gpu` is only exposed in a
 * secure context, so the page has to be served over http://localhost rather
 * than opened as a file or as about:blank.  And Playwright's bundled Chromium
 * does not expose WebGPU at all, so this uses the installed Chrome.
 */

import { chromium } from "@playwright/test";
import { createServer, type ViteDevServer } from "vite";

export interface PageRun<T> {
  readonly value: T;
  readonly consoleErrors: readonly string[];
}

/**
 * Serve the app, open `path`, wait for `globalName` to appear on `window`, and
 * return it.
 */
export async function inChrome<T>(
  path: string,
  globalName: string,
  timeoutMs = 300_000,
): Promise<PageRun<T>> {
  let server: ViteDevServer | undefined;
  let browser: Awaited<ReturnType<typeof chromium.launch>> | undefined;
  try {
    server = await createServer({ configFile: "vite.config.ts", logLevel: "warn" });
    await server.listen();
    const port = server.config.server.port ?? 4173;

    // 🔴 `MIR_CHROME_CHANNEL=chromium` runs Playwright's **bundled** build
    // instead of the installed Google Chrome, which is the only second adapter
    // a one-GPU machine has. It is worth having because every number this
    // project reports comes from one adapter, and the bundled build turns out
    // to expose WebGPU and to answer differently -- the mixed-radix FFT check
    // reads 3.3e-4 relative RMS there against 1.4e-6 on Chrome's Metal path.
    // See docs/measurements/adapter-chromium-bundled-m3pro.json.
    const channel = process.env["MIR_CHROME_CHANNEL"] ?? "chrome";
    browser = await chromium.launch({
      // The bundled build is selected by passing no channel at all.
      ...(channel === "chromium" ? {} : { channel }),
      headless: true,
      args: ["--enable-unsafe-webgpu"],
    });
    const page = await browser.newPage();
    const consoleErrors: string[] = [];
    page.on("console", (m) => {
      if (m.type() === "error") consoleErrors.push(m.text());
    });
    page.on("pageerror", (e) => consoleErrors.push(String(e)));

    await page.goto(`http://localhost:${port}${path}`);
    await page.waitForFunction(
      (name) => (window as unknown as Record<string, unknown>)[name] !== undefined,
      globalName,
      { timeout: timeoutMs },
    );
    const value = (await page.evaluate(
      (name) => (window as unknown as Record<string, unknown>)[name],
      globalName,
    )) as T;
    return { value, consoleErrors };
  } finally {
    await browser?.close();
    await server?.close();
  }
}

/** Fail loudly if the page could not get a device, rather than reporting nothing. */
export function unwrap<T extends object>(value: T | { error: string }): T {
  if ("error" in value) throw new Error(String(value.error));
  return value as T;
}
