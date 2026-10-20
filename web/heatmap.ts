/**
 * Square matrices as images: the 2D RMSD grid and the cross-correlation map.
 *
 * `plot.ts` draws line series; these two analyses are matrices, and a matrix
 * wants a different treatment. Drawn through an `ImageData` at one pixel per
 * cell and then scaled up by the canvas, which is both faster than a rectangle
 * per cell and what gives the smooth look the notebook gets from matplotlib's
 * bicubic interpolation.
 *
 * The two colour scales are not interchangeable, and that is the point of
 * having both. A 2D RMSD is non-negative and reads as a sequential ramp; a
 * correlation runs from -1 to 1 and needs a **diverging** scale centred on
 * zero, or anti-correlated regions look like weakly correlated ones. The
 * notebook uses `PRGn` and `PiYG_r` for exactly this reason.
 */

export interface HeatmapOptions {
  readonly size: number;
  /** `diverging` fixes the midpoint at zero; `sequential` spans the data. */
  readonly scale: "sequential" | "diverging";
  /** Overrides the data range, which a correlation map wants fixed at -1..1. */
  readonly min?: number;
  readonly max?: number;
}

/** Purple-green, the notebook's `PRGn`, for a non-negative matrix. */
function sequential(t: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [64, 0, 75],
    [153, 112, 171],
    [247, 247, 247],
    [90, 174, 97],
    [0, 68, 27],
  ];
  return interpolate(stops, t);
}

/** Pink-yellow-green reversed, the notebook's `PiYG_r`, centred on white. */
function diverging(t: number): [number, number, number] {
  const stops: [number, number, number][] = [
    [77, 146, 33],
    [184, 225, 134],
    [247, 247, 247],
    [241, 182, 218],
    [197, 27, 125],
  ];
  return interpolate(stops, t);
}

function interpolate(
  stops: readonly [number, number, number][],
  t: number,
): [number, number, number] {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - index;
  const a = stops[index] as [number, number, number];
  const b = stops[index + 1] as [number, number, number];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export function drawHeatmap(
  canvas: HTMLCanvasElement,
  matrix: ArrayLike<number>,
  options: HeatmapOptions,
): { readonly min: number; readonly max: number } {
  const { size, scale } = options;
  let low = options.min ?? Infinity;
  let high = options.max ?? -Infinity;
  if (options.min === undefined || options.max === undefined) {
    for (let i = 0; i < size * size; i++) {
      const value = matrix[i] as number;
      if (options.min === undefined && value < low) low = value;
      if (options.max === undefined && value > high) high = value;
    }
  }
  if (scale === "diverging") {
    // Symmetric about zero, so the midpoint colour means "no correlation"
    // rather than "the middle of whatever range this data happened to have".
    const extent = Math.max(Math.abs(low), Math.abs(high));
    low = -extent;
    high = extent;
  }
  const span = high - low || 1;

  const image = new ImageData(size, size);
  const colour = scale === "diverging" ? diverging : sequential;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const [r, g, b] = colour(((matrix[i * size + j] as number) - low) / span);
      // Row 0 at the top, which is how both matplotlib's `imshow` and the
      // notebook's figures orient these.
      const pixel = (i * size + j) * 4;
      image.data[pixel] = r;
      image.data[pixel + 1] = g;
      image.data[pixel + 2] = b;
      image.data[pixel + 3] = 255;
    }
  }

  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  canvas.width = width;
  canvas.height = width;
  const context = canvas.getContext("2d");
  if (context === null) throw new Error("no 2d context for the heatmap");
  // Drawn small then scaled, so the browser's own smoothing stands in for
  // matplotlib's bicubic interpolation.
  const offscreen = document.createElement("canvas");
  offscreen.width = size;
  offscreen.height = size;
  const offContext = offscreen.getContext("2d");
  if (offContext === null) throw new Error("no 2d context for the heatmap buffer");
  offContext.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
  return { min: low, max: high };
}
