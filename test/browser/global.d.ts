import type { DeviceCheckReport } from "../../src/gpu/checks.js";
import type { ProfileComparison } from "../../src/gpu/validate-profile.js";
import type { NeighbourComparison } from "../../src/gpu/validate-neighbours.js";

declare global {
  interface Window {
    __mirReport?: DeviceCheckReport | { error: string };
    __mirProfile?: ProfileComparison;
    __mirNeighbours?: NeighbourComparison | { error: string };
  }
}
