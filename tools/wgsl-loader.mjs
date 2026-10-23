/**
 * Teach plain Node to import `*.wgsl?raw`, the way Vite does.
 *
 * 🔴 `src/gpu/kernels.ts` imports every kernel as `"./kernels/x.wgsl?raw"`.
 * That is a Vite feature: Vite returns the file's text as the default export,
 * and both the browser build and Vitest resolve it because both run through
 * Vite. Plain Node does not, and says
 *
 *     TypeError [ERR_UNKNOWN_FILE_EXTENSION]: Unknown file extension ".wgsl"
 *
 * which is what killed the first attempt at `tools/dawn-probe.ts` in CI before
 * it printed a single line. `vite-node` would have done the job but Vitest 4 no
 * longer ships it, and pulling a package in just for a diagnostic is worse than
 * twenty lines of loader.
 *
 * Used as `node --import ./tools/wgsl-loader.mjs --import tsx <script>`: this
 * hook claims `.wgsl?raw` and tsx claims `.ts`, and they compose because each
 * defers to `next` for anything it does not recognise.
 */

import { registerHooks } from "node:module";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const RAW = ".wgsl?raw";

registerHooks({
  resolve(specifier, context, next) {
    if (!specifier.endsWith(RAW)) return next(specifier, context);
    // Resolve the real file, then put the query back so `load` can see it.
    const resolved = next(specifier.slice(0, -"?raw".length), context);
    return { ...resolved, url: `${resolved.url}?raw`, shortCircuit: true };
  },
  load(url, context, next) {
    if (!url.endsWith(RAW)) return next(url, context);
    const text = readFileSync(fileURLToPath(url.slice(0, -"?raw".length)), "utf8");
    return {
      format: "module",
      // JSON.stringify escapes the backticks, backslashes and newlines that
      // WGSL is full of; a template literal here would not survive them.
      source: `export default ${JSON.stringify(text)};`,
      shortCircuit: true,
    };
  },
});
