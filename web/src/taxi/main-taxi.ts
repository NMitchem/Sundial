import init, { solveMatching } from "../../../crates/sundial-web/pkg/sundial_lp";
import type { MatchResult } from "../../../crates/sundial-web/types-extra";
import { loadPoints, Pt, TaxiPoints } from "./data";
import { MapView } from "./map";
import { greedyAssign, totalMiles } from "./greedy";
import { LineLayer } from "./lines";

const $ = (id: string) => document.getElementById(id)!;
const GREEN = "#46a758";
const RED = "#e5484d";
const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

function animate(ms: number, frame: (t: number) => void): Promise<void> {
  return new Promise((done) => {
    const t0 = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - t0) / ms);
      frame(t);
      if (t < 1) requestAnimationFrame(tick);
      else done();
    };
    requestAnimationFrame(tick);
  });
}

function rollNumber(el: HTMLElement, from: number, to: number, ms: number, suffix: string) {
  return animate(ms, (t) => {
    el.textContent = `${(from + (to - from) * ease(t)).toFixed(0)}${suffix}`;
  });
}

const pairs = (n: number) => (n * (n - 1)) / 2;

class Demo {
  private riders: Pt[];
  private greedy!: Uint32Array;
  private optimal: Uint32Array | null = null;
  private busy = false;
  private pendingTap: Pt | null = null;
  private readonly lines: LineLayer;

  constructor(
    private readonly pts: TaxiPoints,
    private readonly view: MapView,
  ) {
    this.riders = pts.riders.slice();
    this.lines = new LineLayer(view.linesCtx, view.toPx, this.riders, pts.cabs);
  }

  /** Beats 2–3: greedy tangle, then the certified snap. */
  async dispatch() {
    this.busy = true;
    ($("dispatch") as HTMLButtonElement).disabled = true;

    this.greedy = greedyAssign(this.riders, this.pts.cabs);
    const gm = totalMiles(this.riders, this.pts.cabs, this.greedy, this.pts.miles_per_unit);
    $("phase").textContent =
      "The obvious way — every rider grabs the nearest free cab. Look at the mess.";
    await animate(900, (t) =>
      this.lines.draw({ from: this.greedy, reveal: Math.ceil(t * this.riders.length), color: RED }),
    );
    $("miles").textContent = `${gm.toFixed(0)} total pickup miles`;

    const nvars = this.riders.length * this.pts.cabs.length;
    $("phase").textContent =
      `Now the best answer that exists. Searching ${nvars.toLocaleString()} possible routings on your GPU…`;
    let res: MatchResult;
    try {
      res = await this.solve();
    } catch (err) {
      this.fail(err);
      return;
    }

    const { certified, slackFeet } = this.certify(res);
    this.optimal = Uint32Array.from(res.assignment);
    const om = res.total_cost * this.pts.miles_per_unit;
    await animate(900, (t) =>
      this.lines.draw({ from: this.greedy, to: this.optimal!, t: ease(t), color: GREEN }),
    );
    await rollNumber($("miles"), gm, om, 700, " total pickup miles");
    const pct = ((gm - om) / gm) * 100;
    if (certified) {
      const perPickup = slackFeet / this.riders.length;
      $("phase").textContent =
        `${om.toFixed(0)} miles — ${pct.toFixed(0)}% less driving than the obvious way. ` +
        `Proven: no dispatch on Earth beats this by more than ${perPickup.toFixed(1)} ft per pickup.`;
      $("dare").textContent =
        `${pairs(this.riders.length).toLocaleString()} pairs of green routes on screen. Find two that cross. You won't.`;
      $("banner").textContent = "";
    } else {
      $("phase").textContent =
        `Best routing found: ${om.toFixed(0)} miles — ${pct.toFixed(0)}% less than the obvious way.`;
      $("banner").textContent =
        "This run stopped before certification — showing the best routing found, not a proven optimum.";
      $("dare").textContent = "";
    }
    this.receipts(res, certified);
    $("poke-hint").textContent = "Tap anywhere in Manhattan — you need a cab.";
    this.busy = false;
  }

  /** Beat 4: tap → append rider → re-solve → ripple. */
  async poke(p: Pt) {
    if (this.busy) {
      this.pendingTap = p; // queued, not dropped
      return;
    }
    if (this.riders.length >= this.pts.cabs.length) {
      $("poke-hint").textContent = "Cab supply exhausted — every cab in the record is now booked.";
      return;
    }
    this.busy = true;
    const prev = this.optimal!;
    const you = this.riders.length;
    this.riders = [...this.riders, p];
    this.lines.setPoints(this.riders, this.pts.cabs);
    this.lines.setYou(you);
    this.view.renderDots(this.riders, this.pts.cabs, you);
    const nvars = this.riders.length * this.pts.cabs.length;
    $("poke-hint").textContent = `Re-planning the whole city around you — ${nvars.toLocaleString()} routes…`;
    let res: MatchResult;
    try {
      res = await this.solve();
    } catch (err) {
      this.fail(err);
      return;
    }
    const next = Uint32Array.from(res.assignment);
    const changed = new Set<number>();
    for (let i = 0; i < prev.length; i++) if (prev[i] !== next[i]) changed.add(i);
    this.optimal = next;
    const { certified } = this.certify(res);
    await animate(1200, (t) =>
      this.lines.draw({ from: next, color: GREEN, flash: t < 0.8 ? changed : undefined }),
    );
    const om = res.total_cost * this.pts.miles_per_unit;
    $("miles").textContent = `${om.toFixed(0)} total pickup miles`;
    $("poke-hint").textContent =
      `Your cab is on its way. Your single tap re-routed ${changed.size.toLocaleString()} other pickups. Tap again.`;
    this.receipts(res, certified);
    this.busy = false;
    if (this.pendingTap) {
      const q = this.pendingTap;
      this.pendingTap = null;
      void this.poke(q);
    }
  }

  private solve(): Promise<MatchResult> {
    const flat = (ps: Pt[]) => {
      const a = new Float32Array(ps.length * 2);
      ps.forEach((p, i) => {
        a[2 * i] = p[0];
        a[2 * i + 1] = p[1];
      });
      return a;
    };
    return solveMatching(flat(this.riders), flat(this.pts.cabs), 1e-4, (e: any) => {
      $("receipts").textContent =
        `iteration ${e.iter.toLocaleString()} · gap ${e.rel_gap.toExponential(1)} · ${e.ms_per_iter.toFixed(2)} ms/iter`;
    }) as Promise<MatchResult>;
  }

  /** Honesty gate: LP certificate + measured slack over the rigorous floor
   *  (mirrors the native gate's 5e-3 regression bound). */
  private certify(res: MatchResult): { certified: boolean; slackFeet: number } {
    const slackUnits = res.total_cost - res.certified_floor;
    const slackFeet = slackUnits * this.pts.miles_per_unit * 5280;
    const certified =
      res.status.startsWith("Optimal") &&
      slackUnits >= 0 &&
      slackUnits / res.total_cost <= 5e-3;
    return { certified, slackFeet };
  }

  private receipts(res: MatchResult, certified: boolean) {
    const floorMi = res.certified_floor * this.pts.miles_per_unit;
    const { slackFeet } = this.certify(res);
    const base =
      `${res.n_vars.toLocaleString()} routes · ${res.iterations.toLocaleString()} iterations · ` +
      `${(res.solve_ms / 1000).toFixed(1)} s on ${res.adapter} · ` +
      `certified floor ${floorMi.toFixed(2)} mi · slack ${slackFeet.toFixed(0)} ft`;
    $("receipts").textContent = certified
      ? `${base} · LP optimality verified independently on CPU (f64)`
      : `${base} · status: ${res.status}`;
  }

  private fail(err: unknown) {
    console.error(err);
    $("banner").textContent = `GPU solve failed: ${err}. `;
    const retry = document.createElement("button");
    retry.textContent = "Retry";
    retry.onclick = () => {
      $("banner").textContent = "";
      this.busy = false;
      void this.dispatch();
    };
    $("banner").appendChild(retry);
    this.busy = false;
  }

  attach(linesCanvas: HTMLCanvasElement) {
    ($("dispatch") as HTMLButtonElement).onclick = () => void this.dispatch();
    linesCanvas.onclick = (e) => {
      if (!this.optimal) return; // pokes only after the first solve
      const rect = linesCanvas.getBoundingClientRect();
      const [x, y] = this.view.cssToNorm(e.clientX, e.clientY, rect);
      if (x < 0 || y < 0 || y > 1 || x > this.pts.aspect) return;
      void this.poke([x, y]);
    };
  }
}

async function boot() {
  if (!("gpu" in navigator)) {
    ($("nogpu") as HTMLElement).style.display = "block";
    ($("dispatch") as HTMLButtonElement).disabled = true;
    return;
  }
  const pts = await loadPoints();
  const view = new MapView(
    $("base") as HTMLCanvasElement,
    $("dots") as HTMLCanvasElement,
    $("lines") as HTMLCanvasElement,
    pts.aspect,
    Math.min(720, Math.round(window.innerHeight * 0.78)),
  );
  view.renderBasemap(pts.basemap);
  view.renderDots(pts.riders, pts.cabs);
  $("phase").textContent = "Press DISPATCH.";
  await init();
  const demo = new Demo(pts, view);
  demo.attach($("lines") as HTMLCanvasElement);
}

boot().catch((err) => {
  $("phase").textContent = `Failed to load: ${err}`;
  console.error(err);
});
