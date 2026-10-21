/**
 * The validation page: fetch a reference system and compare the kernels
 * against OpenMM on this device.
 *
 * Development only. The fixtures are not committed and a production build has
 * no route to them, so the page says that rather than failing obscurely.
 */

import { parseSystem } from "../src/engine/system.js";
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
import { checksum, decodeNpy } from "../src/analysis/npy.js";
import { validateRandom, type RandomComparison } from "../src/gpu/validate-random.js";
import { validateCellList, type CellListComparison } from "../src/gpu/validate-cell-list.js";
import { validateLoop, type LoopComparison } from "../src/gpu/validate-loop.js";
import {
  validateMinimise,
  type CapturedMinimisation,
  type MinimiseComparison,
} from "../src/gpu/validate-minimise.js";
import {
  validateRestraint,
  type RestraintComparison,
} from "../src/gpu/validate-restraint.js";
import {
  validateBarostat,
  type BarostatComparison,
} from "../src/gpu/validate-barostat.js";
import {
  protocolHooksFor,
  validateProtocol,
  type ProtocolComparison,
} from "../src/gpu/validate-protocol.js";
import { validatePbc, type PbcComparison } from "../src/gpu/validate-pbc.js";
import {
  validateProfile,
  type ProfileComparison,
} from "../src/gpu/validate-profile.js";

import {
  validatePersistence,
  type PersistenceComparison,
} from "../src/gpu/validate-persistence.js";

declare global {
  interface Window {
    __mirValidation?: ValidationReport | { error: string };
    __mirConstraints?: ConstraintComparison | { error: string };
    __mirIntegrators?: IntegratorReport | { error: string };
    __mirRandom?: RandomComparison | { error: string };
    __mirCellList?: CellListComparison | { error: string };
    __mirProfile?: ProfileComparison;
    __mirLoop?: LoopComparison | { error: string };
    __mirMinimise?: MinimiseComparison | { error: string };
    __mirRestraint?: RestraintComparison | { error: string };
    __mirBarostat?: BarostatComparison | { error: string };
    __mirProtocol?: ProtocolComparison | { error: string };
    __mirPbc?: PbcComparison | { error: string };
    __mirPersistence?: PersistenceComparison | { error: string };
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

/** One `.npy` file from the dev server. */
async function fetchNpy(
  name: string,
  file: string,
): Promise<ReturnType<typeof decodeNpy>> {
  const response = await fetch(`/fixtures/${name}/${file}`);
  if (!response.ok) {
    throw new Error(`/fixtures/${name}/${file} returned ${response.status}.`);
  }
  return decodeNpy(await response.arrayBuffer(), file);
}

/** The captured trajectory for one integrator, fetched as three `.npy` files. */
async function fetchTrajectory(
  name: string,
  label: string,
  steps: number,
): Promise<TrajectoryFrames & { checksums: Record<string, number> }> {
  const sums: Record<string, number> = {};
  const load = async (kind: string): Promise<Float64Array> => {
    const file = `trajectory-${label}-${kind}.npy`;
    const response = await fetch(`/fixtures/${name}/${file}`);
    if (!response.ok) {
      throw new Error(
        `/fixtures/${name}/${file} returned ${response.status}. ` +
          "Capture it with tools/capture_constraints.py.",
      );
    }
    const bytes = await response.arrayBuffer();
    // Checksummed because one of these arrived corrupt once: a single wrong f64
    // made a step disagree with OpenMM by 0.0167 nm while still matching the
    // reference, since both had been given the same bad input. Printing the
    // checksum is what separates that from a physics bug.
    sums[`${label}/${kind}`] = checksum(bytes);
    return decodeNpy(bytes, file).data;
  };
  const [positions, velocities, forces] = await Promise.all([
    load("positions"),
    load("velocities"),
    load("forces"),
  ]);
  return { steps, positions, velocities, forces, checksums: sums };
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

function renderPersistence(p: PersistenceComparison): string {
  const total = p.artefacts.reduce((sum, a) => sum + a.bytes, 0);
  return `<h2>Artefacts and resume</h2><p class="note">
      ${p.artefacts.length} artefacts in OPFS, ${(total / 1024).toFixed(1)} kB in
      all, and every one identical to the same run written to memory — byte for
      byte, except that a <code>.log</code> is compared without its ns/day
      column and <code>run.json</code> not at all, since both record wall-clock
      time. That is the test that catches the trap in
      <code>FileSystemWritableFileStream</code>: opening a writable without
      <code>keepExistingData</code> truncates, so patching a DCD header at the
      end of a run would empty the trajectory instead of fixing two words.
    </p>
    <p class="note">
      The device was then destroyed for real
      (${p.deviceWasLost ? "confirmed lost" : "NOT LOST — the test did not do what it claims"}),
      a new one acquired, and the protocol run again over the same directory. It
      skipped <strong>${p.skippedAfterLoss.join(", ") || "nothing"}</strong> and
      redid <strong>${p.redoneAfterLoss.join(", ") || "nothing"}</strong> —
      stage granularity is the notebook's own, so a lost device costs at most
      one stride.
      <span class="badge ${p.ok ? "ok" : "bad"}">${p.ok ? "pass" : "fail"}</span>
    </p>
    ${
      p.differing.length === 0 && p.missing.length === 0
        ? ""
        : `<p class="bad-box">differing: ${p.differing.join(", ")}; missing: ${p.missing.join(", ")}</p>`
    }`;
}

function renderPbc(p: PbcComparison): string {
  return `<h2>Periodic boundaries</h2><p class="note">
      Measured as invariances rather than audited by reading. The box is
      ${p.box.map((v) => v.toFixed(3)).join(" x ")} nm with a ${p.cutoff} nm
      cutoff, so the minimum image is meaningful out to
      ${p.minimumImageLimit.toFixed(3)} nm. The fixture's own coordinates have
      ${p.inputIntact ? "intact" : "SPLIT"} molecules, the longest bonded
      separation being ${p.inputWorstBondedSeparation.toFixed(3)} nm.
    </p>
    <table>
      <tr><th>Change</th><th>Energy (kJ/mol)</th><th>Relative</th>
          <th>Force rel. RMS</th><th></th></tr>
      ${p.invariances
        .map(
          (c) => `<tr>
            <td>${c.what}</td>
            <td class="num">${c.energy.toFixed(3)}</td>
            <td class="num">${exponential(c.energyRelativeError)}</td>
            <td class="num">${exponential(c.forceRelativeRms)}</td>
            <td><span class="badge ${c.ok ? "ok" : "bad"}">${c.ok ? "pass" : "fail"}</span></td>
          </tr>`,
        )
        .join("")}
    </table>
    <p class="note">
      The last row is the complement and must <em>not</em> be invariant: bonded
      terms use plain differences, as OpenMM's do, so splitting a molecule reads
      a bond as a box length. It costs a factor of
      ${p.splitMoleculeEnergyRatio.toFixed(1)} in total energy, which is why
      nothing in the engine ever wraps a coordinate and why an uploaded state is
      checked for intact molecules.
      <span class="badge ${p.ok ? "ok" : "bad"}">${p.ok ? "pass" : "fail"}</span>
    </p>`;
}

function renderProtocol(p: ProtocolComparison): string {
  return `<h2>A whole run</h2><p class="note">
      Minimise with the restraints on, draw velocities, equilibrate under NPT,
      then produce by strides -- a trajectory, a log and a checkpoint for each,
      and <code>run.json</code> for the lot. Short, because a test lane has to
      afford it; the plan's 1 ns gate is this code with bigger numbers.
    </p>
    <p class="note">
      Ensemble <strong>${p.ensemble}</strong>, chosen from the measured cell
      headroom of ${(p.cellHeadroom * 100).toFixed(1)}%: a box narrower than
      three cutoffs cannot use a 27-cell scan, and 1L2Y at the notebook's
      default padding is 3.08 cutoffs wide, so 2.6% of compression exhausts it.
      That is finding 4 in <code>docs/NOTEBOOK_PARITY.md</code> as a hard limit
      rather than a warning.
    </p>
    <table>
      <tr><th>Stage</th><th>Steps</th><th>Frames</th><th>Log rows</th>
          <th>Log steps</th><th>Temperature (K)</th><th>ns/day</th><th></th></tr>
      ${p.stages
        .map(
          (s) => `<tr>
            <td>${s.name}</td>
            <td class="num">${s.steps}</td>
            <td class="num">${s.frames}</td>
            <td class="num">${s.logRows}</td>
            <td class="num">${s.firstLogStep ?? "—"}–${s.lastLogStep ?? "—"}</td>
            <td class="num">${
              s.temperatureRange === null
                ? "—"
                : `${s.temperatureRange[0].toFixed(1)}–${s.temperatureRange[1].toFixed(1)}`
            }</td>
            <td class="num">${s.nsPerDay.toFixed(2)}</td>
            <td><span class="badge ${s.ok ? "ok" : "bad"}">${s.ok ? "pass" : "fail"}</span></td>
          </tr>`,
        )
        .join("")}
    </table>
    <p class="note">
      ${p.artefacts.length} artefacts:
      ${p.artefacts
        .map(
          (a) =>
            `<code>${a.name}</code> ${(a.bytes / 1024).toFixed(1)} kB` +
            (a.frames === null ? "" : ` (${a.frames} frames${a.sizeConsistent ? "" : ", SIZE MISMATCH"})`),
        )
        .join(", ")}.
      A second pass over the same outputs
      ${p.resumeSkippedEverything ? "skipped every stage" : "did NOT skip everything"}.
      <span class="badge ${p.ok ? "ok" : "bad"}">${p.ok ? "pass" : "fail"}</span>
    </p>`;
}

function renderBarostat(b: BarostatComparison): string {
  return `<h2>Barostat</h2><p class="note">
      ${b.molecules.toLocaleString("en-US")} molecules. The move and the
      molecule decomposition are checked in the unit lane; what only runs here
      is whether the box is genuinely dynamic -- whether changing it rewrites
      <em>every</em> uniform carrying a box-derived quantity. A stale one is not
      a crash, it is an energy slightly wrong in a way that looks like ordinary
      single-precision disagreement, so the device's assembled energy is
      checked by changing the box and comparing against a loop <em>constructed</em>
      at that box: the two reach every uniform by different routes, so agreement
      means both wrote the same thing. Asserted exactly -- the same kernels over
      the same data give the same bits.
    </p>
    <table>
      <tr><th>Length scale</th><th>After setBox (kJ/mol)</th><th>After construction</th>
          <th>Difference</th><th>Constraint error change</th><th></th></tr>
      ${b.changes
        .map(
          (c) => `<tr>
            <td class="num">${c.lengthScale}</td>
            <td class="num">${c.afterSetBox.toFixed(4)}</td>
            <td class="num">${c.afterConstruction.toFixed(4)}</td>
            <td class="num">${c.difference}</td>
            <td class="num">${exponential(c.constraintErrorChange)}</td>
            <td><span class="badge ${c.ok ? "ok" : "bad"}">${c.ok ? "pass" : "fail"}</span></td>
          </tr>`,
        )
        .join("")}
    </table>
    <p class="note">
      Over ${b.steps} steps at frequency 25 it made ${b.attempts} attempts and
      accepted ${b.accepted}; the volume went from
      ${b.startVolume.toFixed(2)} to ${b.finalVolume.toFixed(2)} nm&sup3; with a
      proposal width of ${b.volumeScale.toFixed(3)} nm&sup3;.
      <span class="badge ${b.ok ? "ok" : "bad"}">${b.ok ? "pass" : "fail"}</span>
    </p>`;
}

function renderRestraint(r: RestraintComparison): string {
  return `<h2>Position restraints</h2><p class="note">
      ${r.restrainedAtoms.toLocaleString("en-US")} atoms restrained, compared
      against the restraint's own OpenMM force group. The reference
      configuration has exactly zero energy and zero force, so it alone would
      pass a kernel that computed nothing; the wrapped one displaces every atom
      by a whole box, which is what exercises the minimum image.
    </p>
    <table>
      <tr><th>Case</th><th>Device (kJ/mol)</th><th>OpenMM</th><th>Difference</th>
          <th>Worst force error</th><th>vs our f64</th><th>Largest force</th><th></th></tr>
      ${r.cases
        .map(
          (c) => `<tr>
            <td>${c.label}</td>
            <td class="num">${c.deviceEnergy.toFixed(4)}</td>
            <td class="num">${c.openmmEnergy.toFixed(4)}</td>
            <td class="num">${exponential(c.energyError)}</td>
            <td class="num">${exponential(c.worstForceError)}</td>
            <td class="num">${exponential(c.worstForceVsReference)}</td>
            <td class="num">${c.openmmMaxForce.toFixed(2)}</td>
            <td><span class="badge ${c.ok ? "ok" : "bad"}">${c.ok ? "pass" : "fail"}</span></td>
          </tr>`,
        )
        .join("")}
    </table>
    ${
      r.assembled === null
        ? ""
        : `<p class="note">
            Under a rigid ${r.assembled.translation} nm translation -- which
            leaves every distance in the force field untouched while moving every
            restrained atom exactly that far from its reference point -- the
            loop's assembled force field gives
            ${r.assembled.deviceEnergy.toFixed(2)} kJ/mol against the
            double-precision reference's
            ${r.assembled.referenceEnergy.toFixed(2)},
            ${exponential(r.assembled.relativeError)} relative. The restraint
            contributes ${r.assembled.restraintEnergy.toFixed(1)} of it, against
            ${r.assembled.expectedRestraintEnergy.toFixed(1)} in closed form.
            That is what says the restraint pass is encoded into a step rather
            than prepared and forgotten, and ${r.assembled.steps} steps from the
            fixture's own state stayed finite.
            <span class="badge ${r.assembled.ok ? "ok" : "bad"}">${
              r.assembled.ok ? "pass" : "fail"
            }</span>
          </p>`
    }`;
}

function renderMinimise(m: MinimiseComparison): string {
  const row = (frame: MinimiseComparison["converged"]): string => `<tr>
      <td class="num">${frame.iterations}</td>
      <td class="num">${frame.energy.toFixed(4)}</td>
      <td class="num">${frame.openmmEnergy === null ? "—" : frame.openmmEnergy.toFixed(4)}</td>
      <td class="num">${
        frame.energyRelativeError === null
          ? "—"
          : exponential(frame.energyRelativeError)
      }</td>
      <td class="num">${frame.forceRms.toFixed(2)}</td>
      <td class="num">${frame.openmmForceRms === null ? "—" : frame.openmmForceRms.toFixed(2)}</td>
      <td class="num">${exponential(frame.maxConstraintError)}</td>
      <td class="num">${frame.rounds}/${frame.evaluations}</td>
      <td><span class="badge ${frame.ok ? "ok" : "bad"}">${frame.ok ? "pass" : "fail"}</span></td>
    </tr>`;
  return `<h2>Minimisation</h2><p class="note">
      L-BFGS on the host with the forces on the device, which is how OpenMM's own
      minimiser works. Energies compared at <em>matched iteration counts</em>: a
      minimum is not unique, so only a shared path compares tightly. The
      fixture's own state violates its constraints by
      ${(m.startConstraintError * 100).toFixed(1)}% before anything runs.
    </p>
    <table>
      <tr><th>Iterations</th><th>Energy (kJ/mol)</th><th>OpenMM</th>
          <th>Relative</th><th>|f| rms</th><th>OpenMM</th>
          <th>Constraint error</th><th>Rounds/evals</th><th></th></tr>
      ${m.frames.map(row).join("")}
      ${row(m.converged)}
    </table>`;
}

function renderLoop(loop: LoopComparison): string {
  return `<h2>Step loop</h2><p class="note">
      ${loop.passesPerStep} passes a step, one submission per batch, nothing
      allocated or read back in between. Peak pool
      ${(loop.peakPoolBytes / (1024 * 1024)).toFixed(1)} MB.
    </p>
    <table>
      <tr><th>Two runs of 25 steps differ by</th>
          <td class="num">${loop.determinismDifference}</td></tr>
      <tr><th>One step vs the reference, same forces</th>
          <td class="num">${exponential(loop.onePositionVsReference)} nm,
              ${exponential(loop.oneVelocityVsReference)} nm/ps</td></tr>
      <tr><th>Verlet over ${loop.conservationSteps} steps
              (${loop.conservationPicoseconds.toFixed(1)} ps)</th>
          <td class="num">${loop.totalEnergyChange.toFixed(3)} kJ/mol total,
              ${loop.driftPerAtomPerNs.toFixed(4)} per atom per ns</td></tr>
      <tr><th>Potential, start and end</th>
          <td class="num">${loop.firstReport.potentialEnergy.toFixed(2)} →
              ${loop.lastReport.potentialEnergy.toFixed(2)} kJ/mol</td></tr>
      <tr><th>Temperature, start and end</th>
          <td class="num">${loop.firstReport.temperature.toFixed(1)} →
              ${loop.lastReport.temperature.toFixed(1)} K</td></tr>
      <tr><th>Throughput</th>
          <td class="num">${loop.nsPerDay.toFixed(2)} ns/day at 2 fs</td></tr>
      <tr><th></th><td><span class="badge ${loop.ok ? "ok" : "bad"}">
          ${loop.ok ? "pass" : "fail"}</span></td></tr>
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

    // Where a step's GPU time goes. Development-only, like the rest of this
    // page: it needs `timestamp-query` and reports its absence rather than
    // failing when a device does not offer it.
    const profile = await validateProfile(device, { systemXml, stateXml });
    if (body) {
      body.innerHTML += !profile.available
        ? `<div class="check"><h3>Where a step's time goes</h3>
           <p class="note">${profile.reason ?? "unavailable"}</p></div>`
        : `<div class="check"><h3>Where a step's time goes</h3>
        <p class="claim">
          Per-kernel GPU time through <code>timestamp-query</code>. Summed GPU
          time is what the kernels cost; wall-clock is what the caller waited.
          The gap is launch overhead and queue bubbles, and no kernel change
          will move it — which is why both are shown.
        </p>
        <p class="note">
          ${profile.particleCount.toLocaleString("en-US")} particles over
          ${profile.repeats} steps in one submission:
          <strong>${profile.perStepGpuMicroseconds.toFixed(1)} µs</strong> of GPU
          time per step against
          <strong>${profile.perStepWallMicroseconds.toFixed(1)} µs</strong> of
          wall-clock, so ${(profile.overheadShare * 100).toFixed(0)}% of the
          time is outside every kernel. Implied
          ${profile.impliedStepsPerSecond.toFixed(0)} steps/s.
        </p>
        <table><tr><th>Kernel</th><th>Share</th><th>µs/step</th></tr>` +
          profile.kernels
            .map(
              (kernel) =>
                `<tr><td><code>${kernel.label}</code></td>` +
                `<td>${(kernel.share * 100).toFixed(1)}%</td>` +
                `<td>${kernel.microseconds.toFixed(2)}</td></tr>`,
            )
            .join("") +
          "</table></div>";
    }
    window.__mirProfile = profile;

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
      window.__mirLoop = { error: message };
      window.__mirMinimise = { error: message };
      window.__mirBarostat = { error: message };
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

      // The loop starts from the captured frame, not from state.xml: that state
      // is un-minimised (docs/NOTEBOOK_PARITY.md), and an energy-drift number
      // measured from a configuration with a 5.6e5 kJ/mol/nm force on it would
      // mean nothing.
      const components = verlet.positions.length / (dynamics.trajectory_verlet.steps + 1);
      const loop = await validateLoop(device, { systemXml, stateXml }, {
        positions: verlet.positions.slice(0, components),
        velocities: verlet.velocities.slice(0, components),
      });
      if (body) body.innerHTML += renderLoop(loop);
      window.__mirLoop = loop;

      // The minimiser starts from the fixture's own state.xml, which is what a
      // freshly built system looks like -- 25% constraint violation included.
      const minimiseResponse = await fetch(`/fixtures/${name}/minimise-facts.json`);
      const capturedMinimisation = minimiseResponse.ok
        ? ((await minimiseResponse.json()) as CapturedMinimisation)
        : null;
      const minimised = await validateMinimise(
        device,
        { systemXml, stateXml },
        capturedMinimisation,
        name,
      );
      if (body) body.innerHTML += renderMinimise(minimised);
      window.__mirMinimise = minimised;

      // Restraints come from a separate fixture, because the force is added by
      // a run rather than by the force field, so no plain system carries one.
      const restraintFacts = await fetch(`/fixtures/${name}/restraint-facts.json`);
      if (!restraintFacts.ok) {
        const note =
          `/fixtures/${name}/restraint-facts.json is absent; build it with ` +
          "tools/build_restraint_fixture.py.";
        if (body) body.innerHTML += `<h2>Position restraints</h2><p class="note">${note}</p>`;
        window.__mirRestraint = { error: note };
      } else {
        const restraintData = (await restraintFacts.json()) as {
          cases: Record<string, { energy_kJ_mol: number }>;
        };
        const restrainedXml = await fetchText("system-restrained.xml");
        const labels = Object.keys(restraintData.cases);
        const loaded = await Promise.all(
          labels.map(async (label) => ({
            label,
            positions: (await fetchNpy(name, `restraint-${label}-positions.npy`)).data,
            forces: (await fetchNpy(name, `restraint-${label}-forces.npy`)).data,
            energy: restraintData.cases[label]!.energy_kJ_mol,
          })),
        );
        const restraintResult = await validateRestraint(
          device,
          restrainedXml,
          stateXml,
          loaded,
        );
        if (body) body.innerHTML += renderRestraint(restraintResult);
        window.__mirRestraint = restraintResult;
      }

      // From the captured equilibrated frame, for the reason in
      // `validateBarostat`: state.xml is un-minimised and NVE from it diverges.
      // Artefacts in OPFS, and a run that survives losing the GPU. Done before
      // the long checks so a failure here is seen early.
      const shortRun = {
        jobName: `${name}_persist`,
        seed: 0x5eed,
        equilibration: {
          nanoseconds: 0.002,
          timeStep: 0.002,
          temperature: 298,
          pressure: 1,
          trajectoryInterval: 1,
          logInterval: 1,
          minimisationSteps: 20,
          restraintForceConstant: 700,
        },
        production: {
          nanoseconds: 0.002,
          timeStep: 0.002,
          temperature: 298,
          pressure: 1,
          trajectoryInterval: 1,
          logInterval: 1,
          strides: 2,
        },
      };
      const restrainedForRun = restraintFacts.ok
        ? await fetchText("system-restrained.xml")
        : systemXml;
      // It acquires and destroys its own devices: losing the page's would
      // take out every check after it.
      const persistence = await validatePersistence(
        `mir-${name}`,
        shortRun,
        async (forDevice) =>
          (
            await protocolHooksFor(
              forDevice,
              { systemXml, restrainedXml: restrainedForRun, stateXml },
              shortRun,
            )
          ).hooks,
        parseSystem(systemXml),
      );
      if (body) body.innerHTML += renderPersistence(persistence);
      window.__mirPersistence = persistence;

      const pbc = await validatePbc(device, { systemXml, stateXml });
      if (body) body.innerHTML += renderPbc(pbc);
      window.__mirPbc = pbc;

      // A whole run, on the restrained fixture, at a length the lane affords.
      if (restraintFacts.ok) {
        const restrainedXml = await fetchText("system-restrained.xml");
        const protocolResult = await validateProtocol(
          device,
          { systemXml, restrainedXml, stateXml },
          {
            jobName: name,
            seed: 0x5eed,
            equilibration: {
              nanoseconds: 0.004,
              timeStep: 0.002,
              temperature: 298,
              pressure: 1,
              trajectoryInterval: 2,
              logInterval: 1,
              minimisationSteps: 50,
              restraintForceConstant: 700,
            },
            production: {
              nanoseconds: 0.004,
              timeStep: 0.002,
              temperature: 298,
              pressure: 1,
              trajectoryInterval: 2,
              logInterval: 1,
              strides: 2,
            },
          },
        );
        if (body) body.innerHTML += renderProtocol(protocolResult);
        window.__mirProtocol = protocolResult;
      }

      const barostatResult = await validateBarostat(
        device,
        { systemXml, stateXml },
        {
          positions: verlet.positions.slice(0, components),
          velocities: verlet.velocities.slice(0, components),
        },
      );
      if (body) body.innerHTML += renderBarostat(barostatResult);
      window.__mirBarostat = barostatResult;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (body) body.innerHTML = `<p class="bad-box">${message}</p>`;
    window.__mirValidation = { error: message };
    window.__mirConstraints = { error: message };
    window.__mirIntegrators = { error: message };
    window.__mirRandom = { error: message };
    window.__mirCellList = { error: message };
    window.__mirLoop = { error: message };
    window.__mirMinimise = { error: message };
    window.__mirRestraint = { error: message };
    window.__mirBarostat = { error: message };
    window.__mirProtocol = { error: message };
    window.__mirPbc = { error: message };
    window.__mirPersistence = { error: message };
  }
}

void main();
