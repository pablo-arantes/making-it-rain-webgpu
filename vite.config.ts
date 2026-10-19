import { defineConfig } from "vite";

/**
 * The application is `web/`, and it imports the engine straight out of `src/`.
 * WGSL is loaded with `?raw`, so a kernel is a plain file that both the browser
 * build and the Node test lanes read the same way.
 */
export default defineConfig({
  root: "web",
  // GitHub Pages serves from a repository subpath.
  base: process.env.MIR_BASE ?? "/",
  build: {
    outDir: "../dist/web",
    emptyOutDir: true,
    target: "es2023",
    assetsInlineLimit: 0,
  },
  server: { port: 4173, strictPort: true },
  preview: { port: 4173, strictPort: true },
  assetsInclude: ["**/*.wgsl"],
});
