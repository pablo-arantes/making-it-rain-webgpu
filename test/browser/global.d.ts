import type { DeviceCheckReport } from "../../src/gpu/checks.js";

declare global {
  interface Window {
    __mirReport?: DeviceCheckReport | { error: string };
  }
}
