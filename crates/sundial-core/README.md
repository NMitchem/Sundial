# sundial-core

The Sundial solver. Linear programming that runs on **any** GPU (Apple, AMD,
Intel, NVIDIA) as WebGPU compute shaders via [wgpu](https://wgpu.rs), natively
or in a browser tab. No CUDA.

The algorithm is restarted PDHG, the PDLP family, in f32 WGSL. Operators are
matrix-free, so a problem's constraint matrix never has to exist: a
1,048,576-variable optimal-transport instance solves in about **9.4 seconds**
on an Apple M4 Pro without materializing a single matrix entry.

**The GPU never grades its own homework.** `Optimal` is only returned after an
independent CPU-f64 KKT recheck at the exact point being returned.
`Infeasible` and `Unbounded` are only returned after a CPU-f64 Farkas
certificate verifies what the GPU detector nominated. A missed detection is an
honest `IterationLimit`. A false claim is a bug.

```rust
use sundial_core::problem::{CsrMatrix, LpProblem, SolveOptions, SolveStatus};

// A carpenter sells tables at $5 and chairs at $3. A table takes 2 planks and
// 1 hour, a chair takes 1 plank and 2 hours, and there are 10 planks and 8
// hours. Sundial minimizes, so maximizing 5t + 3c means minimizing -5t - 3c.
let a = CsrMatrix {
    n_rows: 2,
    n_cols: 2,
    indptr: vec![0, 2, 4],          // where each row starts
    indices: vec![0, 1, 0, 1],      // which column each value sits in
    values: vec![2.0, 1.0, 1.0, 2.0],
};
let lp = LpProblem::new(
    "carpenter".into(),
    a,
    vec![-5.0, -3.0],               // objective, always minimized
    0.0,                            // constant offset
    vec![f64::NEG_INFINITY; 2],     // rows have no lower limit...
    vec![10.0, 8.0],                // ...and these upper limits
    vec![0.0, 0.0],                 // you can't make negative furniture
    vec![f64::INFINITY; 2],         // no ceiling on quantity
)
.unwrap();

let sol = sundial_core::reference::solve(&lp, &SolveOptions::default(), &mut |_| {});
assert_eq!(sol.status, SolveStatus::Optimal);   // 4 tables, 2 chairs, $26
```

That's the CPU f64 reference path, which needs no GPU and runs anywhere. For
the GPU, swap in `gpu::engine::solve_gpu`, which is async.

The full worked version, with commentary on every argument:

```bash
cargo run --example tiny_lp
```

## Honest limits

Iterate arithmetic is f32, so the headline tolerance is the 1e-4 tier, which is
PDLP's "moderate accuracy." There's no presolve. A CPU simplex solver still
beats this on small LPs, because the GPU only pays off at scale.

Measured Netlib results, with their caveats attached, are in the
[repository README](https://github.com/NMitchem/Sundial).

## License

MIT OR Apache-2.0, at your option.
