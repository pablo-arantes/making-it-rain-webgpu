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
import {
  validateIntegrators,
  type IntegratorReport,
  type TrajectoryFrames,
} from "../src/gpu/validate-integrator.js";
import { decodeNpy } from "../src/analysis/npy.js";
import { validateRandom, type RandomComparison } from "../src/gpu/validate-random.js";
import { validateCellList, type CellListComparison } from "../src/gpu/validate-cell-list.js";

declare global {
  interface Window {
    __mirValidation?: ValidationReport | { error: string };
    __mirConstraints?: ConstraintComparison | { error: string };
    __mirIntegrators?: IntegratorReport | { error: string };
    __mirRandom?: RandomComparison | { error: string };
    __mirCellList?: CellListComparison | { error: string };
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

/** The captured trajectory for one integrator, fetched as three `.npy` files. */
async function fetchTrajectory(
  name: string,
  label: string,
  steps: number,
): Promise<TrajectoryFrames> {
  const load = async (kind: string): Promise<Float64Array> => {
    const file = `trajectory-${label}-${kind}.npy`;
    const response = await fetch(`/fixtures/${name}/${file}`);
    if (!response.ok) {
      throw new Error(
        `/fixtures/${name}/${file} returned ${response.status}. ` +
          "Capture it with tools/capture_constraints.py.",
      );
    }
    return decodeNpy(await response.arrayBuffer(), file).data;
  };
  const [positions, velocities, forces] = await Promise.all([
    load("positions"),
    load("velocities"),
    load("forces"),
  ]);
  return { steps, positions, velocities, forces };
}

function renderIntegrators(report: IntegratorReport): string {
  const rows = report.steps
    .map(
      (s) => `<tr>
        <td>${s.integrator}</td>
        <td class="num">${s.steps}</td>
        <td class="num">${exponential(s.worstPositionVsReference)}</td>
        <td class="num">${exponential(s.worstPositionVsOpenmm)}</td>
        <td class="num">${exponential(s.worstVelocityVsReference)}</td>
        <td class="num">${exponential(s.worstVelocityVsOpenmm)}</td>
        <td><span class="badge ${s.ok ? "ok" : "bad"}">${s.ok ? "pass" : "fail"}</span></td>
      </tr>`,
    )
    .join("");
  const t = report.thermostat;
  const r = report.reductions;
  return `<h2>Integrators</h2><p class="note">
      One step from each captured frame, driven by the forces OpenMM recorded
      there, so nothing from the force kernels leaks in. Position bound
      ${exponential(report.thresholds.position)} nm, velocity bound
      ${exponential(report.thresholds.velocity)} nm/ps.
    </p>
    <table>
      <tr><th>Integrator</th><th>Steps</th>
          <th>Position vs ref (nm)</th><th>vs OpenMM (nm)</th>
          <th>Velocity vs ref</th><th>vs OpenMM</th><th></th></tr>
      ${rows}
    </table>
    <p class="note">
      With friction ${t.friction} ps<sup>-1</sup> the host draws the same
      deviates from the same counters, so the thermostat is compared exactly
      rather than statistically: positions
      ${exponential(t.worstPositionVsReference)} nm and velocities
      ${exponential(t.worstVelocityVsReference)} nm/ps from the reference, against
      a thermostat that moved the velocities by ${t.velocityChange.toFixed(3)} nm/ps.
      Stage by stage: kick ${exponential(t.stages.afterKick)}, RATTLE
      ${exponential(t.stages.afterRattle)}, drift
      ${exponential(t.stages.afterDriftVelocity)} nm/ps and
      ${exponential(t.stages.afterDriftPosition)} nm, SHAKE
      ${exponential(t.stages.afterShake)} nm, finish
      ${exponential(t.stages.afterFinishVelocity)} nm/ps.
      The centre-of-mass reduction lands ${exponential(r.comVelocityError)} nm/ps
      from the reference and the kinetic energy ${exponential(
        r.kineticEnergyRelativeError,
      )} relative.
      <span class="badge ${report.ok ? "ok" : "bad"}">${report.ok ? "pass" : "fail"}</span>
    </p>`;
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

    // The captured trajectories are a separate tool's output, so a fixture can
    // have a system without them. Say which command produces them rather than
    // failing the force checks that already passed.
    const cells = await validateCellList(device, { systemXml, stateXml });
    if (body) {
      body.innerHTML += `<h2>Cell list</h2><p class="note">
        ${cells.divisions.join(" x ")} cells, ${cells.cellCount} in all, built on
        the device from the position buffer with no readback.
        ${cells.assignmentDifferences} atoms assigned differently from the host
        builder and ${cells.populationDifferences} cells populated differently;
        ${cells.orderDifferences.toLocaleString("en-US")} of
        ${cells.particleCount.toLocaleString("en-US")} ordered differently
        <em>within</em> their cell, which the atomics decide and which must not
        matter. It does not: the same kernel fed both lists gives
        ${cells.hostEnergy.toFixed(4)} and ${cells.deviceEnergy.toFixed(4)} kJ/mol
        and a worst force difference of ${cells.forceDifference}, bit for bit,
        because the accumulator is exact.
        <span class="badge ${cells.ok ? "ok" : "bad"}">${cells.ok ? "pass" : "fail"}</span>
      </p>`;
    }
    window.__mirCellList = cells;

    const randomness = await validateRandom(device);
    if (body) {
      body.innerHTML += `<h2>Thermostat noise</h2><p class="note">
        Threefry-2x32-20 over ${randomness.particles.toLocaleString("en-US")}
        particles: ${randomness.bitMismatches} raw-output mismatches against the
        host, worst deviate difference ${exponential(randomness.worstDeviate)}
        (host ${randomness.worstHost.toFixed(6)}, device
        ${randomness.worstDevice.toFixed(6)}, Box-Muller radius
        ${randomness.worstRadius.toExponential(3)}).
        <span class="badge ${randomness.ok ? "ok" : "bad"}">${
          randomness.ok ? "pass" : "fail"
        }</span>
      </p>`;
    }
    window.__mirRandom = randomness;

    const dynamicsResponse = await fetch(`/fixtures/${name}/dynamics-facts.json`);
    if (!dynamicsResponse.ok) {
      const message =
        `/fixtures/${name}/dynamics-facts.json returned ` +
        `${dynamicsResponse.status}; capture it with tools/capture_constraints.py.`;
      if (body) body.innerHTML += `<h2>Integrators</h2><p class="note">${message}</p>`;
      window.__mirIntegrators = { error: message };
    } else {
      const dynamics = (await dynamicsResponse.json()) as {
        trajectory_verlet: { steps: number };
        trajectory_lfmiddle: { steps: number };
      };
      const [verlet, lfmiddle] = await Promise.all([
        fetchTrajectory(name, "verlet", dynamics.trajectory_verlet.steps),
        fetchTrajectory(name, "lfmiddle", dynamics.trajectory_lfmiddle.steps),
      ]);
      const integrators = await validateIntegrators(
        device,
        { systemXml, stateXml },
        { verlet, lfmiddle },
        name,
      );
      if (body) body.innerHTML += renderIntegrators(integrators);
      window.__mirIntegrators = integrators;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (body) body.innerHTML = `<p class="bad-box">${message}</p>`;
    window.__mirValidation = { error: message };
    window.__mirConstraints = { error: message };
    window.__mirIntegrators = { error: message };
    window.__mirRandom = { error: message };
    window.__mirCellList = { error: message };
  }
}

void main();
