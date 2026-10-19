/**
 * The Phase 0 page: report the GPU, then run the precision checks on it.
 *
 * This is the seed of the GPU panel the plan puts in front of every run, and
 * it is also the lane that exercises the kernels on this machine, since no
 * published Dawn build loads on macOS 14.
 */

import { runBench, type BenchReport } from "../src/gpu/bench.js";
import { runDeviceChecks, type DeviceCheckReport } from "../src/gpu/checks.js";
import {
  onDeviceLost,
  requestEngineAdapter,
  requestEngineDevice,
  WebGpuUnavailable,
} from "../src/gpu/device.js";

declare global {
  // Playwright and tools/ read the finished reports off the page.
  interface Window {
    __mirReport?: DeviceCheckReport | { error: string };
    __mirBench?: BenchReport | { error: string };
  }
}

const gpuBody = document.querySelector<HTMLElement>("#gpu-body");
const checksBody = document.querySelector<HTMLElement>("#checks-body");

const integers = new Intl.NumberFormat("en-US");

function row(label: string, value: string): string {
  return `<tr><th>${label}</th><td class="num">${value}</td></tr>`;
}

function renderAdapter(report: DeviceCheckReport): string {
  const a = report.adapter;
  const e = report.sizeEstimate;
  const named = [a.vendor, a.architecture, a.device].filter(Boolean).join(" · ");
  const mib = (bytes: number): string =>
    `${integers.format(Math.round(bytes / 1048576))} MiB`;
  return `<table>
    ${row("Adapter", named || a.description || "<em>not reported</em>")}
    ${a.isFallback ? row("Fallback", "yes — this is a software rasteriser") : ""}
    ${row("Max buffer", mib(a.limits.maxBufferSize ?? 0))}
    ${row("Max storage binding", mib(a.limits.maxStorageBufferBindingSize ?? 0))}
    ${row("Workgroup storage", `${integers.format(a.limits.maxComputeWorkgroupStorageSize ?? 0)} B`)}
    ${row("Invocations per workgroup", integers.format(a.limits.maxComputeInvocationsPerWorkgroup ?? 0))}
    ${row("Optional features", a.features.length > 0 ? a.features.join(", ") : "none")}
    ${row(
      "Estimated maximum system",
      `${integers.format(e.maxAtoms)} atoms (${e.bytesPerAtom} B/atom, limited by ${e.limitedBy})`,
    )}
  </table>
  <p class="note">
    WebGPU does not expose free video memory, so the size estimate comes from the
    adapter's limits and a per-atom model, not from your actual free memory. The
    run loop checks real allocations before starting.
  </p>`;
}

function renderChecks(report: DeviceCheckReport): string {
  return report.checks
    .map(
      (c) => `<div class="check">
        <h3>${c.name}
          <span class="badge ${c.ok ? "ok" : "bad"}">${c.ok ? "pass" : "fail"}</span>
        </h3>
        <p class="claim">${c.claim}</p>
        <details>
          <summary>What it measured</summary>
          <pre>${JSON.stringify(c.detail, null, 1)}</pre>
        </details>
      </div>`,
    )
    .join("");
}

async function main(): Promise<void> {
  try {
    const adapter = await requestEngineAdapter();
    const device = await requestEngineDevice(adapter);
    onDeviceLost(device, (info) => {
      if (gpuBody) {
        gpuBody.innerHTML =
          `<p class="bad-box">The GPU device was lost (${info.reason}): ` +
          `${info.message}. Reload to try again.</p>` + gpuBody.innerHTML;
      }
    });

    const report = await runDeviceChecks(adapter, device);
    if (gpuBody) gpuBody.innerHTML = renderAdapter(report);
    if (checksBody) checksBody.innerHTML = renderChecks(report);
    window.__mirReport = report;

    // `npm run bench` drives this same page with ?bench=1 rather than a second
    // harness, so the benchmark measures exactly what the checks ran on.
    if (new URLSearchParams(location.search).has("bench")) {
      window.__mirBench = await runBench(adapter, device);
    }
  } catch (error) {
    const message =
      error instanceof WebGpuUnavailable
        ? error.message
        : `Unexpected failure: ${String(error)}`;
    if (gpuBody) gpuBody.innerHTML = `<p class="bad-box">${message}</p>`;
    if (checksBody) {
      checksBody.innerHTML =
        `<p class="note">No device, so nothing was checked. ` +
        `The Colab notebooks remain the way to run this protocol without WebGPU: ` +
        `<a href="https://github.com/pablo-arantes/making-it-rain">Making it Rain</a>.</p>`;
    }
    window.__mirReport = { error: message };
    window.__mirBench = { error: message };
  }
}

void main();
