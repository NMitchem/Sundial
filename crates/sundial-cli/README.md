# sundial-cli

Command-line interface for [Sundial](https://github.com/NMitchem/Sundial), a
linear-programming solver that runs on any GPU as WebGPU compute shaders. It
installs a binary named `sundial`.

```bash
cargo install sundial-cli
```

## Commands

```bash
# solve an MPS file, plain or gzipped
sundial solve problem.mps

# generate and solve an optimal-transport instance.
# --grid 32 is 1,048,576 variables, about 9.4 s on an Apple M4 Pro
sundial transport --grid 32

# benchmark every instance in a directory
sundial bench ./instances --out results.csv

# render a results CSV into a markdown table
sundial report results.csv --out report.md
```

Flags worth knowing: `--engine cpu` runs the f64 CPU reference instead of the
GPU, which is how you check a disagreement. `--tol` sets the target relative
KKT tolerance and `--max-iters` caps the iteration count. `--json` prints
machine-readable output.

## Statuses mean what they say

`Optimal` appears only after an independent CPU-f64 KKT recheck at the exact
point being returned. `Infeasible` and `Unbounded` appear only after a CPU-f64
Farkas certificate verifies them. When the solver can't certify anything you
get `IterationLimit`, not a guess.

## License

MIT OR Apache-2.0, at your option.
