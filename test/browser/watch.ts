/**
 * What a long-running page was doing when it died.
 *
 * 🔴 Four runs have ended in
 *
 *     Error: page.waitForFunction: Target page, context or browser has been closed
 *
 * — twice on the 1 ns gate and twice on `bonded.spec.ts`'s 3HTB case — and the
 * cause is still unknown. The reason it stayed unknown is that the message
 * names the *wait*, so it reads as a condition that never came true, and three
 * explanations were offered and withdrawn before anything was measured:
 * a rebuilt `dist/` under the live preview (the dev-server lane fails the same
 * way), display sleep (`caffeinate` made it fail twelve times *sooner*, and
 * `CVDisplayLink ... -6670` appears in runs that pass), and renderer OOM (the
 * heap is flat). See CLAUDE.md.
 *
 * Every one of those came from reading Chrome's stderr instead of the page. So
 * the page is sampled while the wait is in flight, and the sample is printed
 * where the failure will be read. The distinction it buys is the one that
 * matters: **a page that dies with a rising step count or a growing list of
 * finished checks is an environment problem, and one that dies with neither is
 * the engine's.** The instrumented gate passed, which is a datapoint and not a
 * fix — n=1, and polling a tab is a plausible keep-alive, which is exactly the
 * kind of coincidence that produced the three wrong theories above.
 *
 * One probe serves both pages. `validate.html` sets a `__mir*` global per check
 * as it finishes, so the list is its progress; `run.html` has a `<progress>` and
 * a `__mirRun.state`, which are its. Whichever page this is, the other fields
 * come back null rather than absent, so a reader can tell "not applicable" from
 * "stuck at zero".
 */

import type { Page } from "@playwright/test";

export interface PageWatch {
  /** Stop sampling, and print anything the page reported about itself. */
  stop(): void;
}

export function watchPage(page: Page, label: string, everyMs = 60_000): PageWatch {
  const events: string[] = [];
  const stamp = (): string => new Date().toISOString().slice(11, 19);

  // `crash` and `close` are the two that say the page went away rather than
  // merely erred, and neither reaches the console handlers a spec installs.
  page.on("crash", () => events.push(`${stamp()} PAGE CRASHED`));
  page.on("close", () => events.push(`${stamp()} page closed`));
  page.on("pageerror", (error) => events.push(`${stamp()} pageerror: ${String(error)}`));

  const ticker = setInterval(() => {
    void page
      .evaluate(() => ({
        // Each finished check on the validation page leaves one of these.
        finished: Object.keys(window)
          .filter((key) => key.startsWith("__mir"))
          .sort(),
        state: window.__mirRun?.state ?? null,
        steps: document.querySelector("progress")?.getAttribute("value") ?? null,
        heap:
          (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory
            ?.usedJSHeapSize ?? null,
      }))
      .then((sample) => {
        const heap = sample.heap === null ? "?" : (sample.heap / 1048576).toFixed(0);
        const progress =
          sample.state === null && sample.steps === null
            ? ""
            : ` state=${sample.state} steps=${sample.steps}`;
        console.log(
          `  ${stamp()} ${label} heap=${heap}MB${progress} ` +
            `done=${sample.finished.length}[${sample.finished.join(" ")}]`,
        );
      })
      .catch(() => {
        // The page is gone. `events` already says so, and throwing from a
        // timer would replace the real failure with this one.
      });
  }, everyMs);

  return {
    stop(): void {
      clearInterval(ticker);
      if (events.length > 0) {
        console.log(`\n  ${label} page events:\n  ${events.join("\n  ")}`);
      }
    },
  };
}
