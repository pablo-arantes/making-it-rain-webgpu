/**
 * A line chart on a canvas, with no dependencies.
 *
 * A run reports every few seconds for hours, so the plot has to be cheap and it
 * has to stay readable as the series grows. Nothing here is general: it draws
 * one series against step number, autoscales to what it has been given, and
 * redraws whole rather than tracking damage -- a few hundred points at a few
 * frames per minute is not where any time goes.
 *
 * It is a canvas rather than SVG because a long run accumulates thousands of
 * points, and thousands of DOM nodes is a different kind of problem.
 */

export interface PlotSeries {
  readonly label: string;
  readonly unit: string;
  readonly colour: string;
  readonly points: readonly { readonly x: number; readonly y: number }[];
}

const PADDING = { left: 58, right: 10, top: 14, bottom: 26 };

/** Draw one series, autoscaled, with its latest value called out. */
export function drawPlot(canvas: HTMLCanvasElement, series: PlotSeries): void {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  // The backing store follows the device pixel ratio, or the lines are blurry
  // on every laptop made in the last decade.
  if (canvas.width !== Math.round(width * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  if (context === null) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  // Read the page's own colours so the plot follows light and dark mode
  // without a second source of truth.
  const style = getComputedStyle(canvas);
  const ink = style.getPropertyValue("--fg").trim() || "#222";
  const faint = style.getPropertyValue("--muted").trim() || "#999";
  context.font =
    '11px ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace';

  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;
  if (series.points.length === 0) {
    context.fillStyle = faint;
    context.fillText("waiting for the first report…", PADDING.left, PADDING.top + 14);
    return;
  }

  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of series.points) {
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  // A constant series would otherwise divide by zero and draw nothing; give it
  // a band so the flat line is visible and obviously flat.
  if (maxY - minY < 1e-12) {
    const centre = maxY;
    minY = centre - 1;
    maxY = centre + 1;
  }
  const minX = series.points[0]!.x;
  const maxX = Math.max(series.points.at(-1)!.x, minX + 1);

  const sx = (x: number): number =>
    PADDING.left + ((x - minX) / (maxX - minX)) * plotWidth;
  const sy = (y: number): number =>
    PADDING.top + plotHeight - ((y - minY) / (maxY - minY)) * plotHeight;

  // Axes: two lines and three labels. More would be decoration.
  context.strokeStyle = faint;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(PADDING.left, PADDING.top);
  context.lineTo(PADDING.left, PADDING.top + plotHeight);
  context.lineTo(PADDING.left + plotWidth, PADDING.top + plotHeight);
  context.stroke();

  context.fillStyle = faint;
  const label = (value: number): string =>
    Math.abs(value) >= 1e5 || (Math.abs(value) < 1e-2 && value !== 0)
      ? value.toExponential(2)
      : value.toFixed(Math.abs(value) >= 100 ? 0 : 2);
  context.textAlign = "right";
  context.fillText(label(maxY), PADDING.left - 6, PADDING.top + 8);
  context.fillText(label(minY), PADDING.left - 6, PADDING.top + plotHeight);
  context.textAlign = "left";
  context.fillText(`step ${minX}`, PADDING.left, height - 8);
  context.textAlign = "right";
  context.fillText(`${maxX}`, width - PADDING.right, height - 8);

  context.strokeStyle = series.colour;
  context.lineWidth = 1.5;
  context.beginPath();
  for (const [index, point] of series.points.entries()) {
    const x = sx(point.x);
    const y = sy(point.y);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  }
  context.stroke();

  const latest = series.points.at(-1)!;
  context.fillStyle = series.colour;
  context.beginPath();
  context.arc(sx(latest.x), sy(latest.y), 2.5, 0, 2 * Math.PI);
  context.fill();

  context.fillStyle = ink;
  context.textAlign = "left";
  context.fillText(
    `${series.label}  ${label(latest.y)} ${series.unit}`,
    PADDING.left + 4,
    PADDING.top + 10,
  );
}
