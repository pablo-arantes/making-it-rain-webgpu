/**
 * The run log, in OpenMM's `StateDataReporter` CSV format.
 *
 * The notebook writes
 * `StateDataReporter(log_file, nprint, step=True, kineticEnergy=True,
 * potentialEnergy=True, totalEnergy=True, temperature=True, volume=True,
 * speed=True)`, and every downstream analysis cell reads that file back. So the
 * columns are not ours to choose: anything already written against the
 * notebook's logs has to keep working.
 *
 * 🔴 The column order is **not** the order the flags are passed in.
 * `StateDataReporter` emits a fixed sequence regardless, and the notebook's
 * call lists kinetic before potential while the file has potential first. Taken
 * from OpenMM's own `_constructHeaders()` rather than from the call site:
 *
 *     Step, Potential Energy (kJ/mole), Kinetic Energy (kJ/mole),
 *     Total Energy (kJ/mole), Temperature (K), Box Volume (nm^3),
 *     Speed (ns/day)
 *
 * Quoted, comma-separated, with the header line prefixed by `#`.
 */

/** OpenMM's header line for this set of flags, verbatim. */
export const LOG_COLUMNS = [
  "Step",
  "Potential Energy (kJ/mole)",
  "Kinetic Energy (kJ/mole)",
  "Total Energy (kJ/mole)",
  "Temperature (K)",
  "Box Volume (nm^3)",
  "Speed (ns/day)",
] as const;

export interface LogRow {
  readonly step: number;
  /** kJ/mol. */
  readonly potentialEnergy: number;
  readonly kineticEnergy: number;
  readonly totalEnergy: number;
  /** K. */
  readonly temperature: number;
  /** nm^3. */
  readonly volume: number;
  /**
   * ns/day so far.
   *
   * OpenMM leaves this blank on the first row, because it has no elapsed time
   * to divide by yet, and writes `--` there. Reproduced rather than filled with
   * a zero, which a reader would plot.
   */
  readonly speed: number | null;
}

export const logHeader = (): string =>
  `#${LOG_COLUMNS.map((name) => `"${name}"`).join(",")}\n`;

/**
 * One row.
 *
 * Numbers go out through JavaScript's default formatting, which is the shortest
 * decimal that reads back exactly. OpenMM writes repr-style floats too, so the
 * files differ in digits and not in values -- and these are read by `pandas`
 * and `numpy.loadtxt` downstream, which do not care.
 */
export const logRow = (row: LogRow): string =>
  [
    row.step,
    row.potentialEnergy,
    row.kineticEnergy,
    row.totalEnergy,
    row.temperature,
    row.volume,
    row.speed === null ? "--" : row.speed,
  ].join(",") + "\n";
