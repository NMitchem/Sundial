import init, { solveMps, solveMpsBytes } from "../../crates/sundial-web/pkg/sundial_lp";
import { ConvergenceChart, Sample } from "./chart";

const $ = (id: string) => document.getElementById(id)!;
const chart = new ConvergenceChart($("chart") as HTMLCanvasElement, 1e-4);
let wasmReady = false;
let busy = false;

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

const onProgress = (e: Sample & { ms_per_iter: number }) => {
  try {
    chart.push(e);
    $("iter").textContent = `iter ${e.iter.toLocaleString()}`;
    $("msit").textContent = `${e.ms_per_iter.toFixed(3)} ms/iter`;
  } catch (err) {
    console.error("progress callback failed:", err);
  }
};

function addRow(name: string, wallMs: number, res?: any, error?: unknown) {
  ($("results") as HTMLTableElement).hidden = false;
  const tbody = $("results").querySelector("tbody")!;
  const tr = document.createElement("tr");
  const cells = res
    ? [name, res.n_vars.toLocaleString(), res.status, res.objective.toPrecision(8),
       res.iterations.toLocaleString(), wallMs.toFixed(0),
       res.rel_primal.toExponential(1), res.rel_dual.toExponential(1), res.rel_gap.toExponential(1)]
    : [name, "", `Error: ${error}`, "", "", "", "", "", ""];
  for (const c of cells) {
    const td = document.createElement("td");
    td.textContent = String(c);
    tr.appendChild(td);
  }
  tbody.appendChild(tr);
}

async function run(name: string, solve: () => Promise<any>) {
  if (busy) return;
  busy = true;
  ($("solve") as HTMLButtonElement).disabled = true;
  $("status").textContent = "solving…";
  chart.reset(1e-4);
  const t0 = performance.now();
  try {
    await ensureWasm();
    const res = await solve();
    const wall = performance.now() - t0;
    $("gpu").textContent = res.adapter;
    $("status").textContent = res.status;
    addRow(name, wall, res);
  } catch (err) {
    $("status").textContent = "";
    addRow(name, 0, undefined, err);
  } finally {
    busy = false;
    ($("solve") as HTMLButtonElement).disabled = false;
  }
}

$("solve").addEventListener("click", async () => {
  const name = ($("instance") as HTMLSelectElement).value;
  const mps = await (await fetch(`instances/${name}.mps`)).text();
  await run(name, () => solveMps(mps, 1e-4, onProgress));
});

const drop = $("drop");
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", async (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const file = e.dataTransfer?.files?.[0];
  if (!file) return;
  const bytes = new Uint8Array(await file.arrayBuffer());
  await run(file.name.replace(/(\.(mps|gz))+$/, ""), () => solveMpsBytes(bytes, 1e-4, onProgress));
});
