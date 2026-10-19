/**
 * The validation page: fetch a reference system and compare the kernels
 * against OpenMM on this device.
 *
 * Development only. The fixtures are not committed and a production build has
 * no route to them, so the page says that rather than failing obscurely.
 */

import { requestEngineAdapter, requestEngineDevice } from "../src/gpu/device.js";
import { validateBonded, type ValidationReport } from "../src/gpu/validate.js";
import {
  validateConstraints,
  type ConstraintComparison,
} from "../src/gpu/validate-constraints.js";

declare global {
  interface Window {
    __mirValidation?: ValidationReport | { error: string };
    __mirConstraints?: ConstraintComparison | { error: string };
  }
}

const body = document.querySelector<HTMLElement>("#body");
const exponential = (value: number): string => value.toExponential(2);

function render(report: ValidationReport): string {
  const rows = report.comparisons
    .map(
      (c) => `<tr>
        <td>${c.forceType.replace("Force", "")}</td>
        <td class="num">${c.termCount.toLocaleString("en-US")}</td>
        <td class="num">${c.deviceEnergy.toFixed(4)}</td>
        <td class="num">${c.openmmEnergy === null ? "—" : c.openmmEnergy.toFixed(4)}</td>
        <td class="num">${exponential(c.energyRelativeError)}</td>
        <td class="num">${exponential(c.forceRelativeRms)}</td>
        <td><span class="badge ${c.ok ? "ok" : "bad"}">${c.ok ? "pass" : "fail"}</span></td>
      </tr>`,
    )
    .join("");
  return `<p class="note">
      ${report.fixture}, ${report.particleCount.toLocaleString("en-US")} particles,
      built with OpenMM ${report.openmmVersion}. Energy is compared against
      OpenMM's double-precision Reference platform; forces against our own
      double-precision implementation at the same coordinates. Thresholds:
      ${exponential(report.thresholds.energyRelative)} relative energy,
      ${exponential(report.thresholds.forceRelativeRms)} relative force RMS.
    </p>
    <table>
      <tr>
        <th>Force</th><th>Terms</th><th>Device (kJ/mol)</th><th>OpenMM (kJ/mol)</th>
        <th>Energy rel. err</th><th>Force rel. RMS</th><th></th>
      </tr>
      ${rows}
    </table>`;
}

async function main(): Promise<void> {
  const name = new URLSearchParams(location.search).get("fixture") ?? "1L2Y";
  try {
    const fetchText = async (file: string): Promise<string> => {
      const response = await fetch(`/fixtures/${name}/${file}`);
      if (!response.ok) {
        throw new Error(
          `/fixtures/${name}/${file} returned ${response.status}. ` +
            "Generate it with tools/build_reference_system.py, and note that " +
            "only the dev server serves fixtures.",
        );
      }
      return response.text();
    };
    const [systemXml, stateXml, factsJson] = await Promise.all([
      fetchText("system.xml"),
      fetchText("state.xml"),
      fetchText("facts.json"),
    ]);
    const adapter = await requestEngineAdapter();
    const device = await requestEngineDevice(adapter);
    const report = await validateBonded(device, { name, systemXml, stateXml, factsJson });
    if (body) body.innerHTML = render(report);
    window.__mirValidation = report;

    const constraints = await validateConstraints(device, { systemXml, stateXml });
    if (body) {
      body.innerHTML += `<h2>Constraints</h2><p class="note">
        ${constraints.settleClusters.toLocaleString("en-US")} SETTLE clusters and
        ${constraints.iterativeClusters.toLocaleString("en-US")} iterative, largest
        ${constraints.largestCluster.atoms} atoms and
        ${constraints.largestCluster.constraints} constraints. Compared against
        the reference solver at the same coordinates.
      </p>
      <table>
        <tr><th>Worst difference from reference</th>
            <td class="num">${constraints.worstDifference.toExponential(2)} nm</td></tr>
        <tr><th>Device worst relative constraint error</th>
            <td class="num">${constraints.deviceWorstRelativeError.toExponential(2)}</td></tr>
        <tr><th>Reference worst relative constraint error</th>
            <td class="num">${constraints.referenceWorstRelativeError.toExponential(2)}</td></tr>
        <tr><th>Tolerance</th>
            <td class="num">${constraints.tolerance.toExponential(0)}</td></tr>
        <tr><th></th><td><span class="badge ${constraints.ok ? "ok" : "bad"}">
            ${constraints.ok ? "pass" : "fail"}</span></td></tr>
      </table>`;
    }
    window.__mirConstraints = constraints;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (body) body.innerHTML = `<p class="bad-box">${message}</p>`;
    window.__mirValidation = { error: message };
    window.__mirConstraints = { error: message };
  }
}

void main();
