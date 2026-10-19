/**
 * Units.  OpenMM's, everywhere, with no exceptions.
 *
 * Length nm, time ps, mass amu, energy kJ/mol, charge e, temperature K,
 * pressure bar, force kJ/mol/nm, force constant kJ/mol/nm^2.  The notebook's
 * UI speaks angstrom, nanosecond, femtosecond and molar; those conversions
 * happen once, at the UI boundary, and never inside the engine.
 *
 * The one unit the notebook itself gets wrong is the restraint force constant:
 * its slider is labelled "kJ/mol" and `CustomExternalForce('k*periodicdistance(...)^2')`
 * makes it kJ/mol/nm^2.  See docs/NOTEBOOK_PARITY.md.
 */

/** Boltzmann constant in kJ/mol/K, as OpenMM's `MOLAR_GAS_CONSTANT_R`. */
export const BOLTZMANN_KJ_PER_MOL_K = 0.008314462618153242;

/** Coulomb constant in kJ/mol/nm/e^2, as OpenMM's `ONE_4PI_EPS0`. */
export const ONE_4PI_EPS0 = 138.93545764498457;

/** 1 bar in kJ/mol/nm^3, for the barostat's PV term. */
export const BAR_TO_KJ_PER_MOL_NM3 = 0.06022140762081242;

/** Avogadro's number, for converting a box volume to a density. */
export const AVOGADRO = 6.02214076e23;

export const ANGSTROM_PER_NM = 10;
export const FS_PER_PS = 1000;
export const PS_PER_NS = 1000;

export const nmFromAngstrom = (a: number): number => a / ANGSTROM_PER_NM;
export const angstromFromNm = (nm: number): number => nm * ANGSTROM_PER_NM;
export const psFromFs = (fs: number): number => fs / FS_PER_PS;
export const psFromNs = (ns: number): number => ns * PS_PER_NS;

/**
 * kT in kJ/mol.  Used by the barostat's Metropolis criterion and by the
 * velocity initialiser.
 */
export const kT = (temperatureK: number): number =>
  BOLTZMANN_KJ_PER_MOL_K * temperatureK;

/**
 * The Ewald splitting parameter OpenMM derives from a cutoff and an error
 * tolerance, in nm^-1.
 *
 * `NonbondedForceImpl::calcPMEParameters`:
 *   `alpha = (1.0/cutoff)*sqrt(-log(2.0*tol))`
 *
 * For the notebook's 1.0 nm cutoff and `ewaldErrorTolerance = 0.0005` this is
 * 2.628260884878466 nm^-1, which is the value in
 * docs/measurements/3HTB-JZ4-facts.json.
 */
export const ewaldAlpha = (cutoffNm: number, tolerance: number): number =>
  (1 / cutoffNm) * Math.sqrt(-Math.log(2 * tolerance));

/**
 * The *minimum* PME grid dimension along one box vector, as OpenMM computes it:
 *   `ceil(2*alpha*L/(3*pow(tol, 0.2)))`, never below 6.
 *
 * OpenMM's Reference platform uses this number as-is; its CUDA and OpenCL
 * platforms round it up to a size their FFT likes, which is why the notebook's
 * own grid depends on the platform it ran on.  We round with
 * `smoothFftSize` and record both values in `run.json`.
 */
export const pmeMinimumGridSize = (
  alphaPerNm: number,
  boxLengthNm: number,
  tolerance: number,
): number =>
  Math.max(6, Math.ceil((2 * alphaPerNm * boxLengthNm) / (3 * tolerance ** 0.2)));

/**
 * The smallest integer `>= minimum` whose only prime factors are 2, 3, 5 and 7.
 *
 * A mixed-radix FFT needs a smooth size; this is the same family of sizes
 * OpenMM's GPU platforms pick from.  53 becomes 54, 25 stays 25.
 */
export const smoothFftSize = (minimum: number): number => {
  for (let n = Math.max(1, Math.ceil(minimum)); ; n++) {
    let r = n;
    for (const p of [2, 3, 5, 7]) while (r % p === 0) r /= p;
    if (r === 1) return n;
  }
};
