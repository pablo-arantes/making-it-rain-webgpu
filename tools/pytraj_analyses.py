"""The protein analyses as pytraj computes them, which is the plan's oracle.

`tools/build_analysis_fixture.py` computes these in NumPy from formulae written
out by hand.  That shows the implementation matches the stated definitions; it
does not show it matches *cpptraj's*, and cpptraj is what the notebook runs.
The two can differ in ways nothing else would catch -- whether a fluctuation is
measured about an aligned mean, how a covariance is normalised, which way a
correlation is scaled.

So this calls pytraj exactly as the notebook's cells 38 to 46 do, and writes
the answers for comparison.

🔴 **cpptraj works in angstrom.**  Everything here is converted to nm on the
way out, because that is the engine's unit and a factor of ten is the single
easiest way to make two correct implementations look like they disagree.

Needs an environment where pytraj imports.  On this machine that is
`parametrizani_env`; the `ambertools` env has an x86_64 build that will not load
on arm64.

    .../parametrizani_env/bin/python tools/pytraj_analyses.py \\
        --dcd fixtures/analysis/trajectory.dcd --pdb fixtures/1L2Y/system.pdb \\
        --out fixtures/analysis/pytraj.json
"""

import argparse
import json
import pathlib

import numpy as np
import pytraj as pt

ANGSTROM_PER_NM = 10.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcd", required=True, type=pathlib.Path)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()

    traj = pt.iterload(str(args.dcd), str(args.pdb))
    mask = "@CA"
    ca = traj.topology.select(mask)

    # cell 38: Cα RMSD against frame 0.
    rmsd = pt.rmsd(traj, ref=0, mask=mask)

    # cell 40: radius of gyration.
    radgyr = pt.radgyr(traj, mask=mask)

    # cell 42: Cα RMSF. cpptraj's atomicfluct does no fitting, so the notebook
    # is relying on whatever aligned the trajectory before it. Both are
    # recorded: aligned first, which is what the quantity is supposed to mean,
    # and raw, which is what the notebook's call actually computes.
    aligned = pt.superpose(traj[:], ref=0, mask=mask)
    rmsf_aligned = pt.rmsf(aligned, mask)
    rmsf_raw = pt.rmsf(traj, mask)

    # cell 43: 2D RMSD.
    pairwise = pt.pairwise_rmsd(traj, mask=mask)

    # cell 44: PCA, two vectors, fitting onto frame 0.
    projections, (eigenvalues, eigenvectors) = pt.pca(
        traj, mask=mask, n_vecs=2, fit=True, ref=0
    )

    # cell 46: align, then the correlation matrix.
    correl = pt.matrix.correl(aligned, mask)

    def squeeze(values):
        """pytraj returns rmsf as (n, 2): atom index then value."""
        array = np.asarray(values, dtype=float)
        return array[:, 1] if array.ndim == 2 and array.shape[1] == 2 else array.ravel()

    out = {
        "pytraj": pt.__version__,
        "frames": int(traj.n_frames),
        "caIndices": [int(i) for i in ca],
        # Å -> nm for every length.
        "rmsd": (np.asarray(rmsd) / ANGSTROM_PER_NM).tolist(),
        "radiusOfGyration": (np.asarray(radgyr) / ANGSTROM_PER_NM).tolist(),
        "rmsfAligned": (squeeze(rmsf_aligned) / ANGSTROM_PER_NM).tolist(),
        "rmsfRaw": (squeeze(rmsf_raw) / ANGSTROM_PER_NM).tolist(),
        "pairwiseRmsd": (np.asarray(pairwise) / ANGSTROM_PER_NM).flatten().tolist(),
        # nm^2 for an eigenvalue of a coordinate covariance.
        "pcaEigenvalues": (
            np.asarray(eigenvalues) / ANGSTROM_PER_NM**2
        ).tolist(),
        "pcaProjections": (
            np.asarray(projections).T / ANGSTROM_PER_NM
        ).flatten().tolist(),
        # Dimensionless.
        "crossCorrelation": np.asarray(correl).flatten().tolist(),
    }
    args.out.write_text(json.dumps(out))
    print(f"  pytraj {pt.__version__}, {out['frames']} frames, {len(ca)} CA")
    print(f"  rmsd   {min(out['rmsd']):.6f}..{max(out['rmsd']):.6f} nm")
    print(f"  radgyr {np.mean(out['radiusOfGyration']):.6f} nm")
    print(f"  rmsf   aligned max {max(out['rmsfAligned']):.6f} nm, "
          f"raw max {max(out['rmsfRaw']):.6f} nm")
    print(f"  pca    eigenvalues {[round(v, 6) for v in out['pcaEigenvalues'][:3]]} nm^2")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
