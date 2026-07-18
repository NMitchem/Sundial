import { Pt } from "./data";

/** Rider→cab line layer. Endpoints lerp between two assignments (the snap). */
export class LineLayer {
  private you = -1;

  constructor(
    private readonly ctx: CanvasRenderingContext2D,
    private readonly toPx: (p: Pt) => Pt,
    private riders: Pt[],
    private cabs: Pt[],
  ) {}

  setPoints(riders: Pt[], cabs: Pt[]) {
    this.riders = riders;
    this.cabs = cabs;
  }

  setYou(i: number) {
    this.you = i;
  }

  draw(opts: {
    from: ArrayLike<number>;
    to?: ArrayLike<number>;
    t?: number; // 0 = from, 1 = to (eased upstream)
    reveal?: number; // draw only the first `reveal` lines
    color: string;
    flash?: Set<number>;
  }) {
    const { ctx } = this;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    const n = Math.min(opts.reveal ?? this.riders.length, this.riders.length);
    const t = opts.t ?? 0;
    ctx.lineWidth = 1;
    for (let i = 0; i < n; i++) {
      const [rx, ry] = this.toPx(this.riders[i]);
      const a = this.cabs[opts.from[i]];
      const b = opts.to ? this.cabs[opts.to[i]] : a;
      const [ax, ay] = this.toPx(a);
      const [bx, by] = this.toPx(b);
      const ex = ax + (bx - ax) * t;
      const ey = ay + (by - ay) * t;
      if (i === this.you) {
        ctx.strokeStyle = "#4cc2ff";
        ctx.lineWidth = 2;
      } else if (opts.flash?.has(i)) {
        ctx.strokeStyle = "#f5a623";
        ctx.lineWidth = 1.5;
      } else {
        ctx.strokeStyle = opts.color;
        ctx.lineWidth = 1;
      }
      ctx.globalAlpha = i === this.you ? 1 : 0.55;
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(ex, ey);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
}
