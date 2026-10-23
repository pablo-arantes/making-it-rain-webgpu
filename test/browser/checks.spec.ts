import { expect, test } from "@playwright/test";

/**
 * The browser lane, which on macOS is the only lane that reaches the GPU.
 *
 * It loads the page a user would load and asserts the checks it ran, rather
 * than driving the kernels through a bespoke harness, so the thing under test
 * is the thing that ships.
 */
test("the page reports an adapter and passes every precision check", async ({
  page,
}) => {
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(String(error)));

  await page.goto("/");
  await page.waitForFunction(() => window.__mirReport !== undefined, null, {
    timeout: 120_000,
  });
  const report = await page.evaluate(() => window.__mirReport);

  if (report !== undefined && "error" in report) {
    // 🔴 Two different things, and conflating them is what kept this lane
    // behind `continue-on-error` in CI. A browser with no `navigator.gpu` at
    // all is a statement about the environment -- a headless runner without a
    // GPU, or Playwright's bundled Chromium, which exposes none -- and failing
    // on it says nothing about the kernels. A browser that *has* WebGPU and
    // still cannot give us a device is a real result: the limits we ask for
    // were not grantable, or the adapter refused, and that is worth a red tick.
    //
    // The skip is loud on purpose. `test/precision.gpu.test.ts` makes the same
    // point for the Dawn lane: a test that skips silently is indistinguishable
    // from one that passes, which is the failure mode this project cares about
    // most.
    const hasWebGpu = await page.evaluate(() => "gpu" in navigator);
    if (!hasWebGpu) {
      console.log(
        "\n  Browser lane UNAVAILABLE: this browser exposes no navigator.gpu." +
          `\n  The page said: ${report.error}` +
          "\n  WebGPU needs a secure context, so check the page was served over" +
          "\n  http://localhost and that this is Google Chrome rather than" +
          "\n  Playwright's bundled Chromium, which has no WebGPU at all.\n",
      );
      test.skip(true, "this browser exposes no navigator.gpu");
      return;
    }
    throw new Error(
      `WebGPU is present but the page could not get a device: ${report.error}`,
    );
  }
  expect(report).toBeDefined();
  const checked = report!;

  // Report every failing check together: one device quirk often breaks several,
  // and seeing only the first costs a round trip to find that out.
  const failed = checked.checks.filter((c) => !c.ok);
  expect(
    failed.map((c) => `${c.name}\n${JSON.stringify(c.detail, null, 1)}`).join("\n\n"),
  ).toBe("");
  expect(checked.ok).toBe(true);
  expect(checked.checks.length).toBe(7);

  // The adapter panel has to say something real about the machine.
  expect(checked.sizeEstimate.maxAtoms).toBeGreaterThan(10_000);
  // 🔴 `isFallback` is a claim about the *hardware*, not about correctness, and
  // the two environments differ. A developer machine has a real GPU and a
  // fallback there would mean we had measured SwiftShader and called it this
  // GPU -- so it stays an assertion. A hosted runner has no GPU at all, so
  // WebGPU comes from Mesa through software and `isFallback` is true by
  // construction; failing on that would discard the run rather than read it.
  // Measured on CI: every one of the seven checks passed on that adapter, and
  // this line was the only failure.
  if (process.env["CI"] === undefined) {
    expect(checked.adapter.isFallback).toBe(false);
  } else {
    console.log(
      `  CI adapter: ${checked.adapter.vendor} ${checked.adapter.architecture}` +
        ` fallback=${checked.adapter.isFallback}`,
    );
  }
  expect(consoleErrors).toEqual([]);

  // Printed so `npm run test:browser` doubles as the adapter report, and so the
  // diagnostic checks -- which pass whatever they find -- leave a record.
  console.log(
    JSON.stringify(
      {
        adapter: checked.adapter,
        sizeEstimate: checked.sizeEstimate,
        checks: checked.checks.map((c) => ({ name: c.name, detail: c.detail })),
      },
      null,
      1,
    ),
  );
});
