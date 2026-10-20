/**
 * The protein analyses, against independently computed references.
 *
 * `tools/build_analysis_fixture.py` runs 60 frames of Langevin dynamics on
 * 1L2Y with OpenMM and then computes every analysis twice over: the RMSD with
 * MDTraj *and* with a NumPy SVD Kabsch fit, the rest with NumPy from the
 * formulae written out in that file.
 *
 * 🔴 The plan's gate names **pytraj**, which is not installed in any
 * environment on this machine. So this is weaker than the gate asks in one
 * specific way: it shows the implementation matches the stated definitions and
 * an independent implementation of them, not that it matches *cpptraj's*
 * conventions. Where those conventions are ambiguous — cpptraj does no
 * alignment inside `rmsf` or `matrix.correl` — `src/analysis/protein.ts` says
 * what it does instead of guessing. Running the fixture tool somewhere pytraj
 * exists would close the gap.
 *
 * The superposition is the part worth having two implementations of: the
 * reference fits by SVD and this fits by the quaternion method, so agreement
 * is two routes to the same minimum rather than one algorithm compared with
 * itself.
 *
 * Tolerances are set by the trajectory being f32 on disk. A 3 nm coordinate has
 * an f32 ulp near 2.4e-7 nm, so an analysis that sums over 20 atoms and 60
 * frames lands a few parts in 1e-7 from a f64 computation of the same thing.
 * Nothing here is tolerant of a wrong formula: a missing superposition or a
 * covariance normalised by `frames - 1` moves these by percent, not by 1e-7.
 */

import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { readDcd } from "../src/analysis/dcd.js";
import { readPdb } from "../src/analysis/pdb.js";
import {
  caIndices,
  kernelDensity,
  matrixCsv,
  seriesCsv,
  crossCorrelation,
  pairwiseRmsd,
  pca,
  radiusOfGyration,
  rmsdSeries,
  rmsf,
} from "../src/analysis/protein.js";

interface Reference {
  readonly frames: number;
  readonly interval: number;
  readonly caIndices: readonly number[];
  readonly rmsd: readonly number[];
  readonly rmsdMdtraj: readonly number[];
  readonly radiusOfGyration: readonly number[];
  readonly rmsf: readonly number[];
  readonly pairwiseRmsd: readonly number[];
  readonly pcaEigenvalues: readonly number[];
  readonly pcaProjections: readonly number[];
  readonly crossCorrelation: readonly number[];
}

const dcdPath = "fixtures/analysis/trajectory.dcd";
const refPath = "fixtures/analysis/reference.json";
const pdbPath = "fixtures/1L2Y/system.pdb";
const present = existsSync(dcdPath) && existsSync(refPath) && existsSync(pdbPath);

/** The largest absolute difference, reported with its index for a failure. */
function worst(
  ours: ArrayLike<number>,
  theirs: readonly number[],
): { value: number; at: number } {
  expect(ours.length).toBe(theirs.length);
  let value = 0;
  let at = -1;
  for (let i = 0; i < theirs.length; i++) {
    const difference = Math.abs((ours[i] as number) - (theirs[i] as number));
    if (difference > value) {
      value = difference;
      at = i;
    }
  }
  return { value, at };
}

describe.skipIf(!present)("the protein analyses", () => {
  const reference = present
    ? (JSON.parse(readFileSync(refPath, "utf8")) as Reference)
    : undefined;
  const trajectory = present
    ? readDcd(new Uint8Array(readFileSync(dcdPath)))
    : undefined;
  const topology = present ? readPdb(readFileSync(pdbPath, "utf8")) : undefined;
  const ca = present ? caIndices(topology!) : undefined;

  it("reads the trajectory OpenMM wrote", () => {
    expect(trajectory!.frameCount).toBe(reference!.frames);
    expect(trajectory!.interval).toBe(reference!.interval);
    expect(trajectory!.particleCount).toBe(2373);
    expect(trajectory!.periodic).toBe(true);
  });

  it("selects the same Cα atoms", () => {
    // 🔴 Not by name alone. A calcium ion is `CA` in a residue called `CA`, so
    // a name-only selection puts one in the Cα trace and tilts every analysis.
    expect([...ca!]).toEqual([...reference!.caIndices]);
    for (const atom of ca!) {
      expect(topology!.atoms[atom]!.name).toBe("CA");
      expect(topology!.atoms[atom]!.element.toUpperCase()).toBe("C");
    }
  });

  it("computes the Cα RMSD, against both MDTraj and an SVD fit", () => {
    const ours = rmsdSeries(trajectory!, ca!);
    // The quaternion method against NumPy's SVD: two routes to one minimum.
    expect(worst(ours, reference!.rmsd).value).toBeLessThan(5e-7);
    // And against MDTraj, which is an established tool with this definition.
    expect(worst(ours, reference!.rmsdMdtraj).value).toBeLessThan(5e-6);
    // Frame 0 against itself is zero, and the clamp inside `superpose` is what
    // keeps it from being the square root of a small negative.
    expect(ours[0]).toBeCloseTo(0, 12);
    expect(Number.isNaN(ours[0] as number)).toBe(false);
  });

  it("computes the radius of gyration", () => {
    const ours = radiusOfGyration(trajectory!, ca!);
    expect(worst(ours, reference!.radiusOfGyration).value).toBeLessThan(5e-7);
    // A twenty-residue peptide, so sub-nanometre.
    expect(ours[0]).toBeGreaterThan(0.5);
    expect(ours[0]).toBeLessThan(1.5);
  });

  it("computes the per-atom RMSF about the aligned mean", () => {
    const ours = rmsf(trajectory!, ca!);
    expect(ours).toHaveLength(ca!.length);
    expect(worst(ours, reference!.rmsf).value).toBeLessThan(5e-7);
  });

  it("computes the 2D RMSD matrix", () => {
    const ours = pairwiseRmsd(trajectory!, ca!);
    expect(ours).toHaveLength(reference!.frames * reference!.frames);
    expect(worst(ours, reference!.pairwiseRmsd).value).toBeLessThan(5e-7);
    // Symmetric with a zero diagonal, by construction -- asserted because a
    // transposed index would still produce plausible-looking values.
    const n = reference!.frames;
    for (let i = 0; i < n; i++) {
      expect(ours[i * n + i]).toBe(0);
      for (let j = 0; j < n; j++) {
        expect(ours[i * n + j]).toBe(ours[j * n + i]);
      }
    }
  });

  it("computes the principal components and the projections onto them", () => {
    const ours = pca(trajectory!, ca!, 2);
    expect(ours.components).toBe(2);
    expect(ours.eigenvalues).toHaveLength(ca!.length * 3);
    expect(
      worst(ours.eigenvalues.slice(0, 6), reference!.pcaEigenvalues).value,
    ).toBeLessThan(1e-8);
    expect(worst(ours.projections, reference!.pcaProjections).value).toBeLessThan(5e-6);
    // Descending, and the leading components carry most of the variance.
    for (let k = 1; k < 6; k++) {
      expect(ours.eigenvalues[k]!).toBeLessThanOrEqual(ours.eigenvalues[k - 1]!);
    }
    // Six rigid-body degrees of freedom are removed by the alignment, so the
    // last six eigenvalues are zero to round-off. That is a real check on the
    // alignment having happened at all.
    const size = ca!.length * 3;
    for (let k = size - 6; k < size; k++) {
      expect(Math.abs(ours.eigenvalues[k]!)).toBeLessThan(1e-9);
    }
  });

  it("computes the cross-correlation matrix", () => {
    const ours = crossCorrelation(trajectory!, ca!);
    expect(worst(ours, reference!.crossCorrelation).value).toBeLessThan(5e-6);
    const n = ca!.length;
    for (let i = 0; i < n; i++) {
      // Exactly one on the diagonal, and inside [-1, 1] everywhere.
      expect(ours[i * n + i]).toBeCloseTo(1, 12);
      for (let j = 0; j < n; j++) {
        expect(ours[i * n + j]).toBeGreaterThanOrEqual(-1.000001);
        expect(ours[i * n + j]).toBeLessThanOrEqual(1.000001);
        expect(ours[i * n + j]).toBeCloseTo(ours[j * n + i]!, 12);
      }
    }
  });
});

describe.skipIf(!existsSync(refPath))("the distribution and CSV exports", () => {
  const reference = JSON.parse(readFileSync(refPath, "utf8")) as Reference & {
    readonly kde?: { readonly bandwidth: number; readonly x: readonly number[]; readonly y: readonly number[] };
  };

  it("matches scipy's Gaussian KDE at Scott's bandwidth", () => {
    // The notebook draws seaborn KDEs beside the traces, and seaborn's default
    // is Scott's rule. A different bandwidth is a visibly different curve from
    // the same data, so it is matched rather than chosen.
    if (reference.kde === undefined) return;
    const ours = kernelDensity(reference.rmsd, reference.kde.x.length);
    expect(ours.bandwidth).toBeCloseTo(reference.kde.bandwidth, 15);
    expect(worst(ours.x, reference.kde.x).value).toBeLessThan(1e-12);
    expect(worst(ours.y, reference.kde.y).value).toBeLessThan(1e-9);
  });

  it("gives a density that integrates to one", () => {
    // Independent of the reference: a KDE that does not normalise is the usual
    // way to get a plot that looks right and is not a density.
    const { x, y } = kernelDensity(reference.radiusOfGyration, 400);
    let area = 0;
    for (let i = 1; i < x.length; i++) {
      area += 0.5 * ((y[i] as number) + (y[i - 1] as number)) * ((x[i] as number) - (x[i - 1] as number));
    }
    expect(area).toBeCloseTo(1, 3);
  });

  it("survives a constant series without producing NaN", () => {
    // Zero variance means zero bandwidth, and the Gaussian would divide by it.
    // A minimised structure analysed against itself is exactly this case.
    const { y, bandwidth } = kernelDensity([0.5, 0.5, 0.5, 0.5]);
    expect(bandwidth).toBe(0);
    for (const value of y) expect(Number.isFinite(value)).toBe(true);
  });

  it("writes CSV a spreadsheet and pandas can both read", () => {
    const csv = seriesCsv([
      { name: "frame", values: [0, 1, 2] },
      { name: "rmsd_nm", values: [0, 0.1, 0.2] },
    ]);
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("frame,rmsd_nm");
    expect(lines[1]).toBe("0,0");
    expect(lines[2]).toBe("1,0.1");
    expect(lines).toHaveLength(4);
    // Ragged columns are padded rather than truncating the file, because the
    // notebook's own Rg series is one frame shorter than its RMSD series
    // (finding in docs/NOTEBOOK_PARITY.md, cell 40).
    const ragged = seriesCsv([
      { name: "a", values: [1, 2, 3] },
      { name: "b", values: [1] },
    ]);
    expect(ragged.trim().split("\n")).toEqual(["a,b", "1,1", "2,", "3,"]);

    const matrix = matrixCsv([1, 2, 3, 4], 2);
    expect(matrix).toBe("1,2\n3,4\n");
  });
});
