import { Pt } from "./data";

/** Three stacked canvases: static basemap, dots, animated lines. Normalized
 *  coords: y ∈ [0,1] top-down, x ∈ [0, aspect]. */
export class MapView {
  readonly cssHeight: number;
  readonly linesCtx: CanvasRenderingContext2D;
  private readonly s: number; // device px per normalized unit
  private readonly baseCtx: CanvasRenderingContext2D;
  private readonly dotsCtx: CanvasRenderingContext2D;

  constructor(
    base: HTMLCanvasElement,
    dots: HTMLCanvasElement,
    lines: HTMLCanvasElement,
    aspect: number,
    cssHeight: number,
  ) {
    this.cssHeight = cssHeight;
    const cssWidth = Math.round(cssHeight * aspect);
    const dpr = window.devicePixelRatio || 1;
    for (const c of [base, dots, lines]) {
      c.style.width = `${cssWidth}px`;
      c.style.height = `${cssHeight}px`;
      c.width = Math.round(cssWidth * dpr);
      c.height = Math.round(cssHeight * dpr);
    }
    this.s = cssHeight * dpr;
    this.baseCtx = base.getContext("2d")!;
    this.dotsCtx = dots.getContext("2d")!;
    this.linesCtx = lines.getContext("2d")!;
  }

  toPx = ([x, y]: Pt): Pt => [x * this.s, y * this.s];

  /** Click position (CSS px, from getBoundingClientRect) → normalized. */
  cssToNorm(clientX: number, clientY: number, rect: DOMRect): Pt {
    return [(clientX - rect.left) / rect.height, (clientY - rect.top) / rect.height];
  }

  renderBasemap(pts: Pt[]) {
    const ctx = this.baseCtx;
    ctx.fillStyle = "#0b0e14";
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.fillStyle = "rgba(125, 145, 185, 0.10)";
    for (const p of pts) {
      const [px, py] = this.toPx(p);
      ctx.fillRect(px, py, 1.2, 1.2);
    }
  }

  renderDots(riders: Pt[], cabs: Pt[], you?: number) {
    const ctx = this.dotsCtx;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const dot = (p: Pt, r: number, fill: string) => {
      const [px, py] = this.toPx(p);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
    };
    for (const c of cabs) dot(c, 1.6, "#f5d90a");
    riders.forEach((r, i) => dot(r, 1.6, i === you ? "#4cc2ff" : "#e8ecf4"));
    if (you !== undefined) dot(riders[you], 4, "rgba(76,194,255,0.35)");
  }
}
