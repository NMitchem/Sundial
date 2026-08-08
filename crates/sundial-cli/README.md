# sundial-cli

Command-line interface for [Sundial](https://github.com/NMitchem/Sundial), a
linear-programming solver that runs on any GPU as WebGPU compute shaders.
Installs a binary named `sundial`.

```bash
cargo install sundial-cli
```

## Commands

```bash
# solve an MPS file (.mps or .mps.gz)
sundial solve problem.mps

# generate and solve an optimal-transport instance;
# --grid 32 is 1,048,576 variables (~9.4 s on an Apple M4 Pro)
sundial transport --grid 32

# benchmark every instance in a directory
sundial bench ./instances --out results.csv

# render a results CSV into a markdown table
sundial report results.csv --out report.md
```

Useful flags: `--engine cpu` runs the f64 CPU reference instead of the GPU
(handy for checking a disagreement), `--tol` sets the target relative KKT
tolerance, `--max-iters` caps the iteration count, and `--json` prints
machine-readable output.

Statuses are honest: `Optimal` appears only after an independent CPU-f64 KKT
recheck, and `Infeasible`/`Unbounded` only after a CPU-f64 Farkas certificate.
When the solver cannot certify anything, you get `IterationLimit` rather than a
guess.

## License

MIT OR Apache-2.0, at your option.
