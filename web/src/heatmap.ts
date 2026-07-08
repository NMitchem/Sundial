// g×g heatmap drawn at native resolution; CSS `image-rendering: pixelated`
// does the upscaling. Palette reuses the page accents (navy → teal #17bebb
// → yellow #f0c808) so the three panels and the chart read as one system.
const STOPS: [number, number, number][] = [
  [15, 27, 61],
  [23, 190, 187],
  [240, 200, 8],
];

function ramp(t: number): [number, number, number] {
  const x = Math.min(Math.max(t, 0), 1) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(x));
  const f = x - i;
  const a = STOPS[i], b = STOPS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export function drawHeatmap(canvas: HTMLCanvasElement, values: ArrayLike<number>, g: number) {
  canvas.width = g;
  canvas.height = g;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(g, g);
  let max = 0;
  for (let i = 0; i < values.length; i++) max = Math.max(max, values[i]);
  if (max <= 0) max = 1;
  for (let i = 0; i < g * g; i++) {
    const [r, gr, b] = ramp((values[i] ?? 0) / max);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = gr;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}
