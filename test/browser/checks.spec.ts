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
    throw new Error(`the page could not get a device: ${report.error}`);
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
  expect(checked.checks.length).toBe(6);

  // The adapter panel has to say something real about the machine.
  expect(checked.sizeEstimate.maxAtoms).toBeGreaterThan(10_000);
  expect(checked.adapter.isFallback).toBe(false);
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
