# sundial-lp

GPU linear programming in the browser. Restarted PDHG, the PDLP family, running
as WebGPU compute shaders on whatever GPU the visitor already has. No CUDA, no
server, and the data never leaves the tab.

Every `Optimal` is re-verified on the CPU in f64 before the word appears. The
GPU never grades its own homework.

```bash
npm install sundial-lp
```

```js
import init, { solveMps, webgpuAvailable } from "sundial-lp";

await init();
if (!webgpuAvailable()) {
  throw new Error("WebGPU required: Chrome/Edge 113+, Firefox 141+, Safari 26+");
}

// solveMps(mpsText, tolerance, onProgress) resolves once the answer has been
// re-verified in f64 on the CPU. The callback fires on every convergence check.
const result = await solveMps(mpsText, 1e-4, (p) => {
  console.log(p.iter, p.rel_primal, p.rel_dual, p.rel_gap, p.ms_per_iter);
});

console.log(result.status, result.objective);  // "Optimal (CPU f64 verified)"
```

## Honest limits

Iterate arithmetic is f32, so the tier is 1e-4 relative KKT and tighter tiers
are experimental. There's no presolve. CPU simplex beats this on small LPs,
because the GPU only pays off at scale: the demo solves 1,048,576-variable
optimal transport in about 9 seconds on an Apple M4 Pro, at 0.547 ms per
iteration in the tab against 0.59 ms native.

Part of [Sundial](https://github.com/NMitchem/Sundial). The crate directory is
`crates/sundial-web`, and the crate and npm package are both `sundial-lp`.

## License

MIT OR Apache-2.0, at your option.
