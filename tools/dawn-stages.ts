/**
 * The Dawn start-up sequence, named step by step, callable from either runner.
 *
 * 🔴 Shared so the *same* code can run standalone and under Vitest, because
 * that difference is the whole mystery: standalone on `ubuntu-latest` every
 * stage passes and the process exits normally, while
 * `test/precision.gpu.test.ts` segfaults (`THREADS_EXIT=139`) after the adapter
 * warnings and before any test reports. Two copies of the sequence could not
 * settle that; one copy and two callers can.
 *
 * `say` is injected rather than imported so each caller can choose how to emit
 * — and both choose `writeSync`, because a buffered line is lost when the
 * process takes a signal.
 */

export type Say = (message: string) => void;

/** Everything `test/dawn.ts` does, with a line before and after each step. */
export async function acquireDawn(say: Say): Promise<GPUDevice | null> {
  say("importing webgpu");
  const webgpu = (await import("webgpu")) as unknown as {
    create: (flags: string[]) => GPU;
    globals: Record<string, unknown>;
  };
  say("webgpu imported");

  Object.assign(globalThis, webgpu.globals);
  say("globals assigned");

  const gpu = webgpu.create(["enable-dawn-features=allow_unsafe_apis"]);
  say("instance created");

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  say(`requestAdapter returned ${adapter === null ? "null" : "an adapter"}`);
  if (adapter === null) return null;

  const info = adapter.info as unknown as Record<string, unknown> | undefined;
  say(`adapter: ${String(info?.["vendor"])}/${String(info?.["architecture"])}`);

  say("importing device.js");
  const { requestEngineDevice } = await import("../src/gpu/device.js");
  say("device.js imported");

  say("requestEngineDevice (raised limits)");
  const device = await requestEngineDevice(adapter);
  say("device created");

  device.lost
    ?.then((reason) =>
      say(`DEVICE LOST: ${String((reason as unknown as { message?: string }).message)}`),
    )
    .catch(() => {});
  return device;
}
