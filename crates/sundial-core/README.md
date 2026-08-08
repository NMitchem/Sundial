# sundial-core

The Sundial solver: linear programming that runs on **any** GPU — Apple, AMD,
Intel, NVIDIA — as WebGPU compute shaders via [wgpu](https://wgpu.rs), natively
or in a browser tab. No CUDA.

The algorithm is restarted PDHG (the PDLP family) in f32 WGSL, with matrix-free
operators so a problem's constraint matrix never has to exist in memory: a
1,048,576-variable optimal-transport instance solves in about 9.4 s on an Apple
M4 Pro without materializing a single matrix entry.

**The GPU never grades its own homework.** `Optimal` is only returned after an
independent CPU-f64 KKT recheck at the exact point being returned;
`Infeasible` and `Unbounded` only after a CPU-f64 Farkas certificate verifies
the GPU detector's nomination. A missed detection is an honest
`IterationLimit`; a false claim is a bug.

```rust
use sundial_core::problem::{CsrMatrix, LpProblem, SolveOptions, SolveStatus};

// minimize -5a - 3b  subject to  2a + b <= 10,  a + 2b <= 8,  a,b >= 0
let a = CsrMatrix {
    n_rows: 2,
    n_cols: 2,
    indptr: vec![0, 2, 4],
    indices: vec![0, 1, 0, 1],
    values: vec![2.0, 1.0, 1.0, 2.0],
};
let lp = LpProblem::new(
    "carpenter".into(),
    a,
    vec![-5.0, -3.0],                        // objective (minimized)
    0.0,                                     // constant offset
    vec![f64::NEG_INFINITY; 2],              // row lower bounds
    vec![10.0, 8.0],                         // row upper bounds
    vec![0.0, 0.0],                          // column lower bounds
    vec![f64::INFINITY; 2],                  // column upper bounds
)
.unwrap();

let sol = sundial_core::reference::solve(&lp, &SolveOptions::default(), &mut |_| {});
assert_eq!(sol.status, SolveStatus::Optimal);
```

Run the full worked version, with commentary:

```bash
cargo run --example tiny_lp
```

To solve on the GPU instead, swap `reference::solve` for
`gpu::engine::solve_gpu` (async — the example shows both).

## Honest limits

f32 iterate arithmetic, so the headline tolerance is the 1e-4 tier (PDLP's
"moderate accuracy"). No presolve. A CPU simplex solver still wins on small
LPs — the GPU pays off at scale. See the
[repository README](https://github.com/NMitchem/Sundial) for measured Netlib
results with their caveats attached.

## License

MIT OR Apache-2.0, at your option.
