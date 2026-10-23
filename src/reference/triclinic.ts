/**
 * The general triclinic minimum image, in f64, and the box reduction it needs.
 *
 * All three of the notebook's shapes run now -- `cube`, `dodecahedron` and
 * `octahedron` -- in the kernels as well as here, and a 5,002-atom
 * dodecahedron passes the 1 ns gate in 15.3 minutes. A dodecahedron holds
 * 70.7% of the water of the enclosing cube for the same padding, measured here
 * as a determinant ratio and not taken on trust.
 *
 * 🔴 **But not for the same padding**, which is the part deviation 19 had
 * backwards. The 27-cell scan needs three cells an axis across the
 * *perpendicular* widths, and this shape's narrowest is 0.707 of its edge, so
 * it needs `3*sqrt(2)*cutoff` where a cube needs `3*cutoff`. The volume saving
 * is real; the minimum box is larger. `cellListFits` in `src/prep/solvate.ts`
 * is what tells a user that before they run.
 *
 * 🔴 The expensive thing this looked like it would need turns out not to be
 * needed at all -- see `MAXIMUM_CUTOFF_FRACTION`. The minimum image is three
 * subtractions, the same shape as the rectangular path, as long as the cutoff
 * respects a bound that is checked rather than assumed. `src/gpu/` calls this
 * module's reduction through its own WGSL copy of the same arithmetic, and
 * `nearest_image_triclinic` in `fixed64.wgsl` is held to it.
 */

import type { Vec3 } from "../engine/system.js";

export type BoxVectors = readonly [Vec3, Vec3, Vec3];

/**
 * Half a box, per axis, is **not** the right bound for a triclinic cell.
 *
 * The usable cutoff is half the smallest *perpendicular* width -- the distance
 * between opposite faces -- which for a skewed cell is smaller than any edge
 * length. For a 3 nm dodecahedron the edges are 3, 3 and 3.67 nm while the
 * perpendicular widths are 2.12, 2.12 and 3, so the cutoff ceiling is 1.06 nm
 * where the enclosing cube's would be 1.5.
 */
export function perpendicularWidths(box: BoxVectors): Vec3 {
  const [a, b, c] = box;
  const cross = (u: Vec3, v: Vec3): Vec3 => [
    (u[1] as number) * (v[2] as number) - (u[2] as number) * (v[1] as number),
    (u[2] as number) * (v[0] as number) - (u[0] as number) * (v[2] as number),
    (u[0] as number) * (v[1] as number) - (u[1] as number) * (v[0] as number),
  ];
  const norm = (u: Vec3): number =>
    Math.hypot(u[0] as number, u[1] as number, u[2] as number);
  const volume = Math.abs(
    (a[0] as number) * ((b[1] as number) * (c[2] as number) - (b[2] as number) * (c[1] as number)) -
      (a[1] as number) * ((b[0] as number) * (c[2] as number) - (b[2] as number) * (c[0] as number)) +
      (a[2] as number) * ((b[0] as number) * (c[1] as number) - (b[1] as number) * (c[0] as number)),
  );
  // Width across a pair of faces is the volume divided by that face's area.
  return [
    volume / norm(cross(b, c)),
    volume / norm(cross(c, a)),
    volume / norm(cross(a, b)),
  ];
}

/** The cell volume, which is what makes a non-cubic box worth having. */
export function boxVolume(box: BoxVectors): number {
  const [a, b, c] = box;
  return Math.abs(
    (a[0] as number) * ((b[1] as number) * (c[2] as number) - (b[2] as number) * (c[1] as number)) -
      (a[1] as number) * ((b[0] as number) * (c[2] as number) - (b[2] as number) * (c[0] as number)) +
      (a[2] as number) * ((b[0] as number) * (c[1] as number) - (b[1] as number) * (c[0] as number)),
  );
}

export class NotReducedForm extends Error {
  constructor(readonly box: BoxVectors) {
    super(
      "a triclinic box has to be in OpenMM's reduced form before the minimum " +
        "image is meaningful: `a` along x, `b` in the xy plane, positive " +
        "diagonal, and every off-diagonal at most half the diagonal above it. " +
        `These vectors are not: ${JSON.stringify(box)}. ` +
        "`reduceBox` puts them in it, and OpenMM's own `reducePeriodicBoxVectors` " +
        "agrees element for element.",
    );
    this.name = "NotReducedForm";
  }
}

/**
 * OpenMM's reduced form, which is a *precondition* and not a convenience.
 *
 * `a` lies along x, `b` in the xy plane, the diagonal is positive, and each
 * off-diagonal is reduced against the diagonal above it. Every lattice this
 * produces is the same lattice -- only the choice of basis vectors changes --
 * so an energy cannot depend on it, which is what makes the reduction safe to
 * apply to a user's box.
 */
export function reduceBox(box: BoxVectors): BoxVectors {
  const v = box.map((row) => [...row] as number[]);
  // 🔴 Ties to even, not away from zero. An off-diagonal at exactly half the
  // diagonal above it is already reduced, and `Math.round` would take it one
  // whole cell further -- turning the dodecahedron's `c = (1.5, 1.5, 2.12)`
  // into `(-1.5, 1.5, 2.12)`. Both describe the same lattice, so no energy
  // changes, but OpenMM keeps the first and a fixture dumped from OpenMM then
  // fails to match element for element. The same tie rule as
  // `roundTiesToEven` in fixed-point.ts, and for the same kind of reason.
  const reduceBy = (ratio: number): number => {
    const nearest = Math.round(ratio);
    return Math.abs(ratio - Math.trunc(ratio)) === 0.5 && nearest % 2 !== 0
      ? nearest - Math.sign(ratio)
      : nearest;
  };
  const subtract = (target: number, source: number, times: number): void => {
    for (let k = 0; k < 3; k++) {
      (v[target] as number[])[k] =
        ((v[target] as number[])[k] as number) - times * ((v[source] as number[])[k] as number);
    }
  };
  // c against b, then c and b against a: the order matters, because reducing
  // against a can undo a reduction against b otherwise.
  subtract(2, 1, reduceBy(((v[2] as number[])[1] as number) / ((v[1] as number[])[1] as number)));
  subtract(2, 0, reduceBy(((v[2] as number[])[0] as number) / ((v[0] as number[])[0] as number)));
  subtract(1, 0, reduceBy(((v[1] as number[])[0] as number) / ((v[0] as number[])[0] as number)));
  return [
    [(v[0] as number[])[0] as number, (v[0] as number[])[1] as number, (v[0] as number[])[2] as number],
    [(v[1] as number[])[0] as number, (v[1] as number[])[1] as number, (v[1] as number[])[2] as number],
    [(v[2] as number[])[0] as number, (v[2] as number[])[1] as number, (v[2] as number[])[2] as number],
  ];
}

/** Whether a box is already in reduced form, to the tolerance of f64 arithmetic. */
export function isReducedForm(box: BoxVectors, tolerance = 1e-9): boolean {
  const [a, b, c] = box;
  if (Math.abs(a[1] as number) > tolerance || Math.abs(a[2] as number) > tolerance) return false;
  if (Math.abs(b[2] as number) > tolerance) return false;
  if ((a[0] as number) <= 0 || (b[1] as number) <= 0 || (c[2] as number) <= 0) return false;
  return (
    Math.abs(b[0] as number) <= 0.5 * (a[0] as number) + tolerance &&
    Math.abs(c[0] as number) <= 0.5 * (a[0] as number) + tolerance &&
    Math.abs(c[1] as number) <= 0.5 * (b[1] as number) + tolerance
  );
}

/**
 * The cutoff a triclinic cell can carry, as a fraction of its narrowest width.
 *
 * 🔴 **This bound is what makes the minimum image cheap, and it is the whole
 * finding.** The sequential reduction -- subtract `c` by the rounded fractional
 * z, then `b`, then `a`, which is three subtractions and exactly the shape of
 * the rectangular path -- is *not* the true nearest image in general. Measured
 * on 3 nm boxes against a brute-force minimum over 5³ lattice images, it
 * overestimates by up to 1.106 nm on a dodecahedron and 0.952 nm on an
 * octahedron.
 *
 * That looked damning, and the first version of this file concluded a 27-image
 * search was needed and that no kernel could afford one. The conclusion was
 * wrong, because the measurement was taken over all separations rather than the
 * ones that matter. Repeated over 200,000 separations a shape, counting only
 * pairs whose *true* separation is inside a cutoff:
 *
 *   shape          narrowest width   wrong at cutoff 1.0   wrong at 1.3x half
 *   cube                 3.000         0 of 31,080             0 of 179,057
 *   dodecahedron         2.121         0 of 34,992         6,839 of  91,663
 *   octahedron           2.449         0 of 36,929        10,729 of 140,381
 *
 * So the reduction is **exact for every pair inside the cutoff**, provided the
 * cutoff is below half the narrowest perpendicular width -- and it starts
 * failing above it, which is why this is a bound and not a guideline. Every
 * separation it gets wrong is one whose true image is further away than the
 * cutoff, where the pair is rejected either way.
 *
 * The practical consequence for a user: a dodecahedron 3 nm across carries a
 * 1.06 nm cutoff, so the notebook's default 1.0 nm fits, but only just. Padding
 * has to be chosen against the perpendicular width rather than the edge length,
 * which is what `maximumCutoff` is for.
 */
export const MAXIMUM_CUTOFF_FRACTION = 0.5;

/** The largest cutoff this box can carry before the minimum image breaks. */
export function maximumCutoff(box: BoxVectors): number {
  return MAXIMUM_CUTOFF_FRACTION * Math.min(...perpendicularWidths(box));
}

export class CutoffTooLargeForBox extends Error {
  constructor(readonly box: BoxVectors, readonly cutoff: number) {
    const widths = perpendicularWidths(box);
    super(
      `a ${cutoff} nm cutoff does not fit this box: its narrowest ` +
        `perpendicular width is ${Math.min(...widths).toFixed(4)} nm, so the ` +
        `minimum image is only exact below ${maximumCutoff(box).toFixed(4)} nm. ` +
        "The edge lengths are longer than the perpendicular widths for a skewed " +
        "cell, so an edge-based bound would pass here and lose pairs. Solvate " +
        "with more padding or shorten the cutoff.",
    );
    this.name = "CutoffTooLargeForBox";
  }
}

/**
 * The nearest image of `to - from` for a reduced triclinic box.
 *
 * Three subtractions, and exact for any separation shorter than
 * `maximumCutoff(box)` -- see `MAXIMUM_CUTOFF_FRACTION` for the measurement
 * that says so and for what happens beyond it. A caller that needs the true
 * nearest image for an arbitrary separation wants `nearestImageBySearch`, which
 * is the oracle this is tested against and is far too slow for a force loop.
 */
export function minimumImage(from: Vec3, to: Vec3, box: BoxVectors): Vec3 {
  if (!isReducedForm(box)) throw new NotReducedForm(box);
  const [a, b, c] = box;
  let x = (to[0] as number) - (from[0] as number);
  let y = (to[1] as number) - (from[1] as number);
  let z = (to[2] as number) - (from[2] as number);

  const kc = Math.round(z / (c[2] as number));
  x -= kc * (c[0] as number);
  y -= kc * (c[1] as number);
  z -= kc * (c[2] as number);
  const kb = Math.round(y / (b[1] as number));
  x -= kb * (b[0] as number);
  y -= kb * (b[1] as number);
  x -= Math.round(x / (a[0] as number)) * (a[0] as number);

  return [x, y, z];
}

/**
 * The true nearest image, by searching the lattice. The oracle, not the path.
 *
 * 🔴 Reduces **first** and searches around that, rather than searching around
 * the raw separation. Searching the raw separation needs a span that grows with
 * how far apart the two points are -- a 6 nm separation in a 3 nm box needs 3
 * where 2 silently returns the wrong image -- so an oracle written that way is
 * only correct for inputs small enough, which is the opposite of what an oracle
 * is for. After the reduction every offset that can arise is within one cell
 * per axis: measured over 20,000 separations a shape, 9 distinct offsets for the
 * dodecahedron and 13 for the octahedron, all of them in {-1, 0, 1}³. So a span
 * of 1 is enough for any input, and the test pins that by checking 2 changes
 * nothing.
 */
export function nearestImageBySearch(
  from: Vec3,
  to: Vec3,
  box: BoxVectors,
  span = 1,
): Vec3 {
  const [a, b, c] = box;
  const reduced = minimumImage(from, to, box);
  const dx = reduced[0] as number;
  const dy = reduced[1] as number;
  const dz = reduced[2] as number;
  let best: Vec3 = [dx, dy, dz];
  let bestSquared = Infinity;
  for (let i = -span; i <= span; i++) {
    for (let j = -span; j <= span; j++) {
      for (let k = -span; k <= span; k++) {
        const x = dx + i * (a[0] as number) + j * (b[0] as number) + k * (c[0] as number);
        const y = dy + j * (b[1] as number) + k * (c[1] as number);
        const z = dz + k * (c[2] as number);
        const squared = x * x + y * y + z * z;
        if (squared < bestSquared) {
          bestSquared = squared;
          best = [x, y, z];
        }
      }
    }
  }
  return best;
}

/** The three box shapes `Modeller.addSolvent` builds, for a given width. */
export function boxVectorsForShape(
  shape: "cube" | "dodecahedron" | "octahedron",
  width: number,
): BoxVectors {
  switch (shape) {
    case "cube":
      return [[width, 0, 0], [0, width, 0], [0, 0, width]];
    case "dodecahedron":
      return [
        [width, 0, 0],
        [0, width, 0],
        [width / 2, width / 2, (width * Math.SQRT2) / 2],
      ];
    case "octahedron":
      return [
        [width, 0, 0],
        [width / 3, (width * 2 * Math.SQRT2) / 3, 0],
        [-width / 3, (width * Math.SQRT2) / 3, (width * Math.sqrt(6)) / 3],
      ];
  }
}

/**
 * The inverse box, as the matrix `M` with `f = r · M`.
 *
 * A point `r` is `fa·a + fb·b + fc·c`, so with the box vectors as the rows of
 * `B`, `r = f·B` and `f = r·B⁻¹`. For a reduced box `B` is lower triangular, so
 * the inverse comes out of back-substitution rather than a general solve and is
 * lower triangular too.
 *
 * Fractional coordinates are what make a triclinic PME possible at all: the
 * charge grid is laid out in `f`, so the FFT is the same transform whatever the
 * cell shape, and only the reciprocal vectors in the convolution change. The
 * alternative -- resampling a skewed grid -- is not a thing anyone does.
 */
export function inverseBox(box: BoxVectors): readonly [Vec3, Vec3, Vec3] {
  if (!isReducedForm(box)) throw new NotReducedForm(box);
  const ax = (box[0] as Vec3)[0] as number;
  const bx = (box[1] as Vec3)[0] as number;
  const by = (box[1] as Vec3)[1] as number;
  const cx = (box[2] as Vec3)[0] as number;
  const cy = (box[2] as Vec3)[1] as number;
  const cz = (box[2] as Vec3)[2] as number;
  // f_c = r_z / cz
  // f_b = (r_y - f_c cy) / by
  // f_a = (r_x - f_b bx - f_c cx) / ax
  return [
    [1 / ax, 0, 0],
    [-bx / (ax * by), 1 / by, 0],
    [((bx * cy) / (by * cz) - cx / cz) / ax, -cy / (by * cz), 1 / cz],
  ];
}

/** `f = r · M`, the fractional coordinate of a Cartesian point. */
export function fractionalOf(
  x: number,
  y: number,
  z: number,
  inverse: readonly [Vec3, Vec3, Vec3],
): Vec3 {
  return [
    x * ((inverse[0] as Vec3)[0] as number) +
      y * ((inverse[1] as Vec3)[0] as number) +
      z * ((inverse[2] as Vec3)[0] as number),
    x * ((inverse[0] as Vec3)[1] as number) +
      y * ((inverse[1] as Vec3)[1] as number) +
      z * ((inverse[2] as Vec3)[1] as number),
    x * ((inverse[0] as Vec3)[2] as number) +
      y * ((inverse[1] as Vec3)[2] as number) +
      z * ((inverse[2] as Vec3)[2] as number),
  ];
}

/**
 * The reciprocal-lattice vector for mesh indices `(ma, mb, mc)`, in nm⁻¹.
 *
 * `g_i · a_j = 2π δ_ij`, so the reciprocal vectors are the rows of `2π B⁻ᵀ` and
 * `k = m · 2π B⁻ᵀ`. For a rectangular box that collapses to `2π m_u / L_u` per
 * axis, which is what the rectangular code computes — so this is a
 * generalisation and not a replacement, and the rectangular results are
 * unchanged bit for bit.
 */
export function reciprocalVector(
  ma: number,
  mb: number,
  mc: number,
  inverse: readonly [Vec3, Vec3, Vec3],
): Vec3 {
  const twoPi = 2 * Math.PI;
  return [
    twoPi * ma * ((inverse[0] as Vec3)[0] as number),
    twoPi *
      (ma * ((inverse[1] as Vec3)[0] as number) +
        mb * ((inverse[1] as Vec3)[1] as number)),
    twoPi *
      (ma * ((inverse[2] as Vec3)[0] as number) +
        mb * ((inverse[2] as Vec3)[1] as number) +
        mc * ((inverse[2] as Vec3)[2] as number)),
  ];
}

/** Whether a box is rectangular, in which case the simpler paths all apply. */
export function isRectangular(box: BoxVectors, tolerance = 1e-12): boolean {
  return (
    Math.abs((box[0] as Vec3)[1] as number) <= tolerance &&
    Math.abs((box[0] as Vec3)[2] as number) <= tolerance &&
    Math.abs((box[1] as Vec3)[0] as number) <= tolerance &&
    Math.abs((box[1] as Vec3)[2] as number) <= tolerance &&
    Math.abs((box[2] as Vec3)[0] as number) <= tolerance &&
    Math.abs((box[2] as Vec3)[1] as number) <= tolerance
  );
}
