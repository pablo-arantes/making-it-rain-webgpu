/** WGSL kernels are loaded as source text, by the browser build and the tests alike. */
declare module "*.wgsl?raw" {
  const source: string;
  export default source;
}
