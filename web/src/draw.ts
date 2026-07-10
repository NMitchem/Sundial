// Brush painting for draw-your-own transport masses. Grid-resolution mass
// arrays (g×g, row-major idx = row*g + col) painted via pointer events with
// a small gaussian splat; heatmaps re-render through the caller's onPaint.
export class DrawState {
  src: Float32Array;
  tgt: Float32Array;
  readonly g: number;
  constructor(g: number) {
    this.g = g;
    this.src = new Float32Array(g * g);
    this.tgt = new Float32Array(g * g);
  }
  clear(side: "src" | "tgt") {
    (side === "src" ? this.src : this.tgt).fill(0);
  }
  /** Paint a gaussian splat (σ ≈ 0.9 cells, radius 3) centered at grid coords. */
  paint(side: "src" | "tgt", gx: number, gy: number) {
    const a = side === "src" ? this.src : this.tgt;
    const g = this.g;
    const cx = Math.floor(gx), cy = Math.floor(gy);
    for (let dy = -3; dy <= 3; dy++) {
      for (let dx = -3; dx <= 3; dx++) {
        const x = cx + dx, y = cy + dy;
        if (x < 0 || y < 0 || x >= g || y >= g) continue;
        const d2 = (x + 0.5 - gx) ** 2 + (y + 0.5 - gy) ** 2;
        const i = y * g + x;
        a[i] = Math.min(10, a[i] + Math.exp(-d2 / (2 * 0.9 * 0.9)));
      }
    }
  }
}

/** Wire pointer painting onto a heatmap canvas (CSS-scaled, native g×g). */
export function attachBrush(
  canvas: HTMLCanvasElement,
  side: "src" | "tgt",
  state: () => DrawState | null,
  onPaint: () => void,
): void {
  let down = false;
  const toGrid = (e: PointerEvent): [number, number] | null => {
    const st = state();
    if (!st) return null;
    const r = canvas.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * st.g, ((e.clientY - r.top) / r.height) * st.g];
  };
  const apply = (e: PointerEvent) => {
    const st = state();
    const p = toGrid(e);
    if (!st || !p) return;
    st.paint(side, p[0], p[1]);
    onPaint();
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (!state()) return;
    down = true;
    canvas.setPointerCapture(e.pointerId);
    apply(e);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (down) apply(e);
  });
  const up = () => {
    down = false;
  };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);
}
