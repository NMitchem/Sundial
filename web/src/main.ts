import init, { transportPreview, solveTransport, solveTransportCustom } from "../../crates/sundial-web/pkg/sundial_lp";
import { ConvergenceChart, Sample } from "./chart";
import { drawHeatmap } from "./heatmap";
import { DrawState, attachBrush } from "./draw";

const $ = (id: string) => document.getElementById(id)!;
const chart = new ConvergenceChart($("chart") as HTMLCanvasElement, 1e-4);
let wasmReady = false;
let drawState: DrawState | null = null;

if (!("gpu" in navigator)) {
  $("nogpu").hidden = false;
  ($("solve") as HTMLButtonElement).disabled = true;
}

async function ensureWasm() {
  if (!wasmReady) {
    await init();
    wasmReady = true;
  }
}

function params() {
  return {
    g: parseInt(($("grid") as HTMLSelectElement).value, 10),
    preset: ($("preset") as HTMLSelectElement).value,
  };
}

function mode(): "preset" | "draw" {
  return ($("mode") as HTMLSelectElement).value as "preset" | "draw";
}

function repaintDraw() {
  if (!drawState) return;
  drawHeatmap($("src") as HTMLCanvasElement, drawState.src, drawState.g);
  drawHeatmap($("tgt") as HTMLCanvasElement, drawState.tgt, drawState.g);
}

async function preview() {
  try {
    await ensureWasm();
    const { g, preset } = params();
    const p = transportPreview(g, preset);
    drawHeatmap($("src") as HTMLCanvasElement, p.src, g);
    drawHeatmap($("tgt") as HTMLCanvasElement, p.tgt, g);
    drawHeatmap($("arriving") as HTMLCanvasElement, new Float32Array(g * g), g);
    $("nvars").textContent = `${(g * g * g * g).toLocaleString()} variables`;
  } catch (err) {
    console.error("preview failed:", err);
  }
}

function enterMode() {
  const isDraw = mode() === "draw";
  ($("preset") as HTMLSelectElement).disabled = isDraw;
  ($("clear-src") as HTMLButtonElement).hidden = !isDraw;
  ($("clear-tgt") as HTMLButtonElement).hidden = !isDraw;
  if (isDraw) {
    const { g } = params();
    drawState = new DrawState(g);
    repaintDraw();
    drawHeatmap($("arriving") as HTMLCanvasElement, new Float32Array(g * g), g);
  } else {
    drawState = null;
    void preview();
  }
}

$("preset").addEventListener("change", () => void preview());
$("grid").addEventListener("change", () => {
  if (mode() === "draw") {
    enterMode();
    return;
  }
  void preview();
});
$("mode").addEventListener("change", enterMode);
$("clear-src").addEventListener("click", () => {
  drawState?.clear("src");
  repaintDraw();
});
$("clear-tgt").addEventListener("click", () => {
  drawState?.clear("tgt");
  repaintDraw();
});
attachBrush($("src") as HTMLCanvasElement, "src", () => drawState, repaintDraw);
attachBrush($("tgt") as HTMLCanvasElement, "tgt", () => drawState, repaintDraw);
void preview();

$("solve").addEventListener("click", async () => {
  const btn = $("solve") as HTMLButtonElement;
  btn.disabled = true;
  $("result").textContent = "";
  $("status").textContent = "loading wasm…";
  try {
    await ensureWasm();
    const { g, preset } = params();
    const ns = g * g;
    chart.reset(1e-4);
    $("status").textContent = "solving…";
    const t0 = performance.now();
    const progressCb = (e: Sample & { ms_per_iter: number }) => {
      try {
        chart.push(e);
        $("iter").textContent = `iter ${e.iter.toLocaleString()}`;
        $("msit").textContent = `${e.ms_per_iter.toFixed(3)} ms/iter`;
      } catch (err) {
        console.error("progress callback failed:", err);
      }
    };
    const snapshotCb = (_iter: number, ax: Float32Array) => {
      try {
        drawHeatmap($("arriving") as HTMLCanvasElement, ax.subarray(ns), g);
      } catch (err) {
        console.error("snapshot callback failed:", err);
      }
    };
    const res = mode() === "draw" && drawState
      ? await solveTransportCustom(g, drawState.src, drawState.tgt, 1e-4, progressCb, snapshotCb)
      : await solveTransport(g, preset, 1e-4, progressCb, snapshotCb);
    $("gpu").textContent = res.adapter;
    $("status").textContent = res.status;
    $("result").textContent =
      `objective ${res.objective.toPrecision(6)} · ${res.n_vars.toLocaleString()} variables · ` +
      `${res.iterations.toLocaleString()} iterations · ${(performance.now() - t0).toFixed(0)} ms wall · ` +
      `verified residuals: primal ${res.rel_primal.toExponential(1)}, ` +
      `dual ${res.rel_dual.toExponential(1)}, gap ${res.rel_gap.toExponential(1)}`;
  } catch (err) {
    $("status").textContent = "";
    $("result").textContent = `Error: ${err}`;
  } finally {
    btn.disabled = false;
  }
});
