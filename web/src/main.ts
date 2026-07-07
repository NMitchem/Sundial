import init, { solveMps } from "../../crates/sundial-mps/pkg/sundial_mps";
import { ConvergenceChart, Sample } from "./chart";

const $ = (id: string) => document.getElementById(id)!;
const chart = new ConvergenceChart($("chart") as HTMLCanvasElement, 1e-4);

if (!("gpu" in navigator)) {
  $("nogpu").hidden = false;
  ($("solve") as HTMLButtonElement).disabled = true;
}

$("solve").addEventListener("click", async () => {
  const btn = $("solve") as HTMLButtonElement;
  btn.disabled = true;
  $("result").textContent = "";
  $("status").textContent = "loading wasm…";
  try {
    await init();
    const name = ($("instance") as HTMLSelectElement).value;
    const mps = await (await fetch(`instances/${name}.mps`)).text();
    chart.reset(1e-4);
    $("status").textContent = "solving…";
    const t0 = performance.now();
    const res = await solveMps(mps, 1e-4, (e: Sample & { ms_per_iter: number }) => {
      chart.push(e);
      $("iter").textContent = `iter ${e.iter.toLocaleString()}`;
      $("msit").textContent = `${e.ms_per_iter.toFixed(3)} ms/iter`;
    });
    $("gpu").textContent = res.adapter;
    $("status").textContent = res.status;
    $("result").textContent =
      `objective ${res.objective.toPrecision(10)} · ${res.iterations.toLocaleString()} iterations · ` +
      `${(performance.now() - t0).toFixed(0)} ms wall · verified residuals: primal ${res.rel_primal.toExponential(1)}, ` +
      `dual ${res.rel_dual.toExponential(1)}, gap ${res.rel_gap.toExponential(1)}`;
  } catch (err) {
    $("status").textContent = "";
    $("result").textContent = `Error: ${err}`;
  } finally {
    btn.disabled = false;
  }
});
