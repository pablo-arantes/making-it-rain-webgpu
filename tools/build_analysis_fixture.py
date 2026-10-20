"""A short trajectory and the analyses computed from it, as a reference.

The protein analyses (Cα RMSD, radius of gyration, RMSF, 2D RMSD, PCA,
cross-correlation) need a trajectory with enough frames to be meaningful, and
numbers computed by something other than the code under test.

🔴 The plan's gate names **pytraj**, and pytraj is not installed in any
environment on this machine (`tools/README.md` says so for the trajectory
readers too).  So the oracle here is MDTraj for RMSD -- an independent
implementation of a standard definition -- and NumPy for the rest, with every
formula written out below rather than taken from a library whose conventions
would then need reverse-engineering.  That is weaker than the gate asks for in
one specific way: it does not prove agreement with *cpptraj's* conventions,
only with the definitions stated here.  Running this where pytraj exists would
close it.

    python tools/build_analysis_fixture.py --system fixtures/1L2Y/system.xml \\
        --state fixtures/1L2Y/state.xml --pdb fixtures/1L2Y/system.pdb \\
        --out fixtures/analysis
"""

import argparse
import json
import pathlib

import numpy as np
import openmm
import openmm.app as app
import openmm.version
from openmm import unit


def kabsch_rmsd(mobile: np.ndarray, reference: np.ndarray) -> tuple[float, np.ndarray]:
    """Best-fit RMSD and the rotation, by SVD.

    A different route from the quaternion method the TypeScript uses, which is
    the point: two implementations of the same minimisation agreeing is
    evidence, two copies of one algorithm agreeing is not.
    """
    m = mobile - mobile.mean(axis=0)
    r = reference - reference.mean(axis=0)
    covariance = m.T @ r
    u, _, vt = np.linalg.svd(covariance)
    # Reflection guard: a negative determinant would give a mirror image, which
    # fits better and is not a rotation.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T
    fitted = m @ rotation.T
    return float(np.sqrt(((fitted - r) ** 2).sum() / len(m))), rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, type=pathlib.Path)
    parser.add_argument("--state", required=True, type=pathlib.Path)
    parser.add_argument("--pdb", required=True, type=pathlib.Path)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--interval", type=int, default=250, help="steps per frame")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    system = openmm.XmlSerializer.deserialize(args.system.read_text())
    state = openmm.XmlSerializer.deserialize(args.state.read_text())
    pdb = app.PDBFile(str(args.pdb))

    # A plain Langevin run at 298 K. The trajectory only has to move; what is
    # being tested is the analysis, not the dynamics.
    integrator = openmm.LangevinMiddleIntegrator(
        298 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds
    )
    integrator.setRandomNumberSeed(24301)
    # Driven through a Simulation so the DCDReporter writes exactly the file
    # OpenMM's own users get, rather than one assembled by hand here.
    dcd = args.out / "trajectory.dcd"
    simulation = app.Simulation(
        pdb.topology, system, integrator, openmm.Platform.getPlatformByName("CPU")
    )
    simulation.context.setState(state)
    simulation.context.setVelocitiesToTemperature(298 * unit.kelvin, 24301)
    simulation.reporters.append(app.DCDReporter(str(dcd), args.interval))
    simulation.step(args.interval * args.frames)

    # --- Reference analyses, over the Cα atoms --------------------------
    import mdtraj

    traj = mdtraj.load_dcd(str(dcd), top=str(args.pdb))
    ca = [a.index for a in traj.topology.atoms if a.name == "CA" and a.element.symbol == "C"]
    xyz = np.asarray(traj.xyz, dtype=np.float64)[:, ca, :]  # nm
    frames = xyz.shape[0]

    # RMSD against frame 0. Two sources: our own Kabsch, and MDTraj's.
    rmsd = np.array([kabsch_rmsd(xyz[t], xyz[0])[0] for t in range(frames)])
    rmsd_mdtraj = mdtraj.rmsd(traj, traj, frame=0, atom_indices=np.asarray(ca))

    # Radius of gyration: sqrt(mean |r - <r>|^2), unweighted (every Cα is carbon).
    centred = xyz - xyz.mean(axis=1, keepdims=True)
    radgyr = np.sqrt((centred**2).sum(axis=2).mean(axis=1))

    # Align every frame onto frame 0 for the three analyses that need a mean.
    aligned = np.empty_like(xyz)
    reference = xyz[0] - xyz[0].mean(axis=0)
    for t in range(frames):
        m = xyz[t] - xyz[t].mean(axis=0)
        _, rotation = kabsch_rmsd(xyz[t], xyz[0])
        aligned[t] = m @ rotation.T
    mean = aligned.mean(axis=0)

    # RMSF about the mean structure.
    rmsf = np.sqrt(((aligned - mean) ** 2).sum(axis=2).mean(axis=0))

    # Pairwise RMSD.
    pairwise = np.zeros((frames, frames))
    for i in range(frames):
        for j in range(i + 1, frames):
            value = kabsch_rmsd(xyz[i], xyz[j])[0]
            pairwise[i, j] = pairwise[j, i] = value

    # PCA of the aligned coordinates, normalised by the frame count.
    flat = (aligned - mean).reshape(frames, -1)
    covariance = (flat.T @ flat) / frames
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    for k in range(vectors.shape[1]):
        j = np.argmax(np.abs(vectors[:, k]))
        if vectors[j, k] < 0:
            vectors[:, k] *= -1
    projections = flat @ vectors[:, :2]

    # Dynamic cross-correlation.
    displacement = aligned - mean
    dot = np.einsum("tix,tjx->ij", displacement, displacement)
    diagonal = np.sqrt(np.diag(dot))
    correl = dot / np.outer(diagonal, diagonal)

    # The KDE seaborn draws beside each trace (cells 39, 41, 45), with scipy's
    # Scott bandwidth so the curve is comparable rather than merely similar.
    from scipy.stats import gaussian_kde

    sd = rmsd.std(ddof=1)
    bandwidth = sd * len(rmsd) ** (-1 / 5)
    low, high = rmsd.min() - 3 * bandwidth, rmsd.max() + 3 * bandwidth
    grid = np.linspace(low, high, 200)
    density = gaussian_kde(rmsd, bw_method=bandwidth / sd)(grid)

    out = {
        "kde": {
            "bandwidth": float(bandwidth),
            "x": grid.tolist(),
            "y": density.tolist(),
        },
        "openmm": openmm.version.version,
        "pdb": args.pdb.name,
        "frames": frames,
        "interval": args.interval,
        "caIndices": [int(i) for i in ca],
        "rmsd": rmsd.tolist(),
        "rmsdMdtraj": [float(v) for v in rmsd_mdtraj],
        "radiusOfGyration": radgyr.tolist(),
        "rmsf": rmsf.tolist(),
        "pairwiseRmsd": pairwise.flatten().tolist(),
        "pcaEigenvalues": values[:6].tolist(),
        "pcaProjections": projections.flatten().tolist(),
        "crossCorrelation": correl.flatten().tolist(),
        "oracle": "MDTraj for RMSD, NumPy for the rest; pytraj is unavailable here",
    }
    (args.out / "reference.json").write_text(json.dumps(out))
    print(f"  {frames} frames, {len(ca)} CA atoms -> {dcd}")
    print(f"  RMSD ours vs MDTraj: max |d| {np.abs(rmsd - rmsd_mdtraj).max():.3e} nm")
    print(f"  RMSD range {rmsd.min():.4f}..{rmsd.max():.4f} nm, Rg {radgyr.mean():.4f} nm")
    print(f"  wrote {args.out / 'reference.json'}")


if __name__ == "__main__":
    main()
