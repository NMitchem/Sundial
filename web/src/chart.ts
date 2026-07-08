export interface Sample { iter: number; rel_primal: number; rel_dual: number; rel_gap: number; }

const SERIES: { key: keyof Omit<Sample, "iter">; color: string; label: string }[] = [
  { key: "rel_primal", color: "#e4572e", label: "primal residual" },
  { key: "rel_dual", color: "#17bebb", label: "dual residual" },
  { key: "rel_gap", color: "#76b041", label: "duality gap" },
];

export class ConvergenceChart {
  private data: Sample[] = [];
  private raf = 0;
  constructor(private canvas: HTMLCanvasElement, private tol: number) {}

  push(s: Sample) {
    this.data.push(s);
    if (this.raf === 0) {
      this.raf = requestAnimationFrame(() => {
        this.raf = 0;
        this.draw();
      });
    }
  }

  reset(tol: number) {
    if (this.raf !== 0) {
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }
    this.data = [];
    this.tol = tol;
    this.draw();
  }

  private draw() {
    const ctx = this.canvas.getContext("2d")!;
    const { width: W, height: H } = this.canvas;
    ctx.clearRect(0, 0, W, H);
    if (this.data.length < 2) return;
    const maxIter = this.data[this.data.length - 1].iter;
    const yMin = Math.min(1e-8, this.tol / 10), yMax = 10;
    const x = (it: number) => 60 + (W - 80) * (it / maxIter);
    const y = (v: number) => {
      const c = Math.min(Math.max(v, yMin), yMax);
      return 20 + (H - 60) * (1 - (Math.log10(c) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)));
    };
    // tolerance line + decade gridlines
    ctx.strokeStyle = "#8886"; ctx.fillStyle = "#888"; ctx.font = "20px system-ui";
    for (let e = Math.ceil(Math.log10(yMin)); e <= 1; e++) {
      const yy = y(10 ** e);
      ctx.beginPath(); ctx.moveTo(60, yy); ctx.lineTo(W - 20, yy); ctx.stroke();
      ctx.fillText(`1e${e}`, 4, yy + 6);
    }
    ctx.strokeStyle = "#f0c808"; ctx.setLineDash([8, 6]);
    ctx.beginPath(); ctx.moveTo(60, y(this.tol)); ctx.lineTo(W - 20, y(this.tol)); ctx.stroke();
    ctx.setLineDash([]);
    // series
    for (const s of SERIES) {
      ctx.strokeStyle = s.color; ctx.lineWidth = 3; ctx.beginPath();
      this.data.forEach((d, i) => {
        const px = x(d.iter), py = y(d[s.key]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }
    // legend
    SERIES.forEach((s, i) => {
      ctx.fillStyle = s.color;
      ctx.fillRect(W - 300, 30 + i * 28, 18, 6);
      ctx.fillText(s.label, W - 275, 40 + i * 28);
    });
  }
}
