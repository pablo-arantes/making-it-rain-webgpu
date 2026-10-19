/**
 * What the order of the minimum-image wrap costs a pair vector.
 *
 * `delta_periodic` in accumulate.wgsl differences two positions exactly in
 * fixed point -- which is the whole reason positions are integers -- and then
 * converts the result to f32 and wraps it. The conversion happens while the
 * value can still be as large as the box diagonal, so it is rounded at the ulp
 * of a 6.58 nm number, 4.8e-7 nm, and the wrap that follows cannot recover
 * what the conversion dropped. Wrapping first and converting the small result
 * instead keeps it.
 *
 * This exists because 3HTB's direct-space force relative RMS is 1.13e-5 against
 * 1L2Y's 7.65e-7 -- thirteen times, for the same kernel -- and the first
 * explanation offered for it (cancellation in the per-atom sum) turned out to
 * be wrong: making that sum exact did not move the number at all. This measures
 * a cause that is real, on the host, in f64, with no GPU involved.
 *
 * 🔴 It is kept after the fix because of how its number was once misused. The
 * ratio below is 2 to 3 for the *vectors*, and that was taken to bound the
 * force gap at "about 3 of the 13" -- which assumed a pair-vector error passes
 * through to a force at unit gain. It does not: the gain is `dF/dr`, around
 * 1e5 kJ/mol/nm^2 for Lennard-Jones at 0.25 nm. Moving the wrap into fixed
 * point improved 3HTB's direct force relative RMS by a factor of **21**, from
 * 1.13e-5 to 5.37e-7, closing the gap entirely. The lesson this test now
 * carries is that an intermediate quantity bounds a final one only if you know
 * the gain between them.
 */

import { describe, expect, it } from "vitest";
import { fixture } from "./fixtures.js";
import { rectangularBox } from "../src/reference/forces/nonbonded.js";
import { toFixed, fromFixed } from "../src/reference/fixed-point.js";
for (const name of ["1L2Y", "3HTB"] as const) {
  const f = fixture(name);
  describe.skipIf(f === null)(name, () => {
    it("measures the pair-vector error the kernel's ordering produces", () => {
      const box = rectangularBox(f!.state.boxVectors);
      const x = f!.state.positions;
      const n = f!.system.particleCount;
      // The kernel: difference exactly in fixed point, convert to f32, then
      // wrap in f32.  The exact version wraps before converting.
      let worstKernel = 0;
      let worstExact = 0;
      let sumSquaredKernel = 0;
      let count = 0;
      // Sample pairs across the whole box so the large raw differences appear.
      for (let i = 0; i < n; i += 97) {
        for (let j = i + 1; j < n; j += 89) {
          for (let axis = 0; axis < 3; axis++) {
            const extent = box[axis] as number;
            const exactRaw = (x[j * 3 + axis] as number) - (x[i * 3 + axis] as number);
            const exactWrapped = exactRaw - extent * Math.round(exactRaw / extent);

            // Fixed point differences exactly; f32 then rounds the *raw* value.
            const fixedRaw = fromFixed(
              toFixed(x[j * 3 + axis] as number) - toFixed(x[i * 3 + axis] as number),
            );
            const asF32 = Math.fround(fixedRaw);
            const kernelWrapped = Math.fround(
              asF32 - Math.fround(extent * Math.round(asF32 / extent)),
            );
            // Wrapping first, then converting.
            const exactThenF32 = Math.fround(
              fixedRaw - extent * Math.round(fixedRaw / extent),
            );

            worstKernel = Math.max(worstKernel, Math.abs(kernelWrapped - exactWrapped));
            worstExact = Math.max(worstExact, Math.abs(exactThenF32 - exactWrapped));
            sumSquaredKernel += (kernelWrapped - exactWrapped) ** 2;
            count++;
          }
        }
      }
      const rmsKernel = Math.sqrt(sumSquaredKernel / count);
      // Reported on stderr rather than asserted against a bound: this is a
      // measurement of a known shortfall, and a test that failed when it got
      // better would be worse than no test.
      process.stderr.write(
        `\n${name}: box ${box.map((v) => (v as number).toFixed(2)).join(" x ")} nm, ` +
          `${count} components sampled\n` +
          `  convert then wrap (the kernel): worst ${worstKernel.toExponential(2)} nm, ` +
          `rms ${rmsKernel.toExponential(2)} nm\n` +
          `  wrap then convert:              worst ${worstExact.toExponential(2)} nm\n`,
      );
      expect(count).toBeGreaterThan(0);
    });
  });
}
