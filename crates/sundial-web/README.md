# sundial-lp

GPU linear programming in the browser: restarted PDHG (the PDLP algorithm
family) as WebGPU compute shaders. No CUDA, no server — and every `Optimal`
is re-verified on the CPU in f64: the GPU never grades its own homework.

```js
import init, { solveMps, webgpuAvailable } from "sundial-lp";

await init();
if (!webgpuAvailable()) throw new Error("WebGPU required (Chrome/Edge 113+, Firefox 141+, Safari 26+)");
const result = await solveMps(mpsText, 1e-4, (p) => console.log(p.iter, p.rel_gap));
console.log(result.status, result.objective); // "Optimal (CPU f64 verified)", …
```

Honest limits: f32 iterate arithmetic at the 1e-4 relative-KKT tier (tighter
tiers experimental); no presolve; CPU simplex beats this on small LPs — the
GPU pays off at scale (the demo solves 1,048,576-variable optimal transport
in ~9 s on an Apple M4 Pro).
