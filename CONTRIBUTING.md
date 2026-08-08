# Contributing to Sundial

Thanks for looking. Sundial is a small, opinionated project — a linear-programming
solver (restarted PDHG, the PDLP family) that runs as WebGPU compute shaders. This
file covers what you need to build it, what the CI gates are, and the one class of
change that will be rejected on principle no matter how well it is written.

## Before you start

Read [`docs/STATUS.md`](docs/STATUS.md) first — it is the current state of the
project, including what is deliberately unfinished and why. The design documents
behind the architecture live in [`docs/design/`](docs/design/).

For anything larger than a bug fix, **open an issue before writing code.** Several
things that look like obvious improvements (double-double accumulation, GPU
presolve, a movement-based primal-weight update) have already been investigated
and deliberately deferred, with the reasoning written down in `docs/notes/`. An
issue first saves you from re-running a closed experiment.

## Building and testing

```bash
cargo test --workspace                        # CPU suite — this is what CI runs
cargo test --workspace -- --include-ignored   # + GPU suite; needs a real GPU
```

GPU tests are `#[ignore]`d so CI (GitHub runners, no GPU) stays green. **If you
touch anything under `crates/sundial-core/gpu/` — including the WGSL shaders — you
must run the ignored tests locally and say so in the PR.** Run the 1M-variable
transport gate in `--release`; it is slow otherwise.

The web demo:

```bash
wasm-pack build crates/sundial-web --target web
cd web && npm install && npm run dev
cd web && npx tsc --noEmit && npm run build
```

Note the naming quirk: the directory is `crates/sundial-web`, but the crate and
npm package are both named **`sundial-lp`** — so it is `cargo build -p sundial-lp`.

Full fresh-clone verification, which is what a release runs:

```bash
bash scripts/verify_clean_checkout.sh
```

## CI gates

CI fails on any of these, so run them before pushing:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build -p sundial-lp --target wasm32-unknown-unknown
cargo clippy -p sundial-lp --target wasm32-unknown-unknown -- -D warnings
```

## Invariants — please do not weaken these

These are the design commitments the project is built on. A PR that relaxes one
will be declined even if it makes the numbers look better.

- **Certificate honesty.** `Optimal` is only ever set after an independent CPU-f64
  KKT recheck at the exact returned point. `Infeasible` and `Unbounded` are only
  ever set after a CPU-f64 Farkas certificate verifies the GPU detector's
  nomination; a failed verification resets the streak. The GPU never grades its own
  homework. A missed detection — an honest `IterationLimit` — is acceptable. A
  false status claim is not. Current infeasibility recall is 2 of 6 on Netlib's
  infeasible set; **do not "fix" that by loosening verification.**
- **GPU infinity convention.** ±1e30 sentinels in every GPU buffer. Shaders never
  do infinity arithmetic — browser fast-math portability depends on this.
- **Grid-stride kernels everywhere.** Dispatch is capped at 4,096 workgroups;
  correctness beyond that comes from stride loops, not larger dispatches.
- **No precision features built on error-free transforms.** wgpu hardcodes
  `fastMathEnabled=YES` on Metal with no control surface, so df64 and Neumaier
  compensation silently collapse to plain f32 there. Read
  [`docs/notes/df64-experiment.md`](docs/notes/df64-experiment.md) before touching
  accumulation code — this is a closed experiment, not open future work.
- **Primal-weight ω applies only to the matrix-free/unscaled path.** The explicit
  Ruiz+Pock–Chambolle path runs at τ=σ, because ω limit-cycles there.
- **Timing convention.** `solve_ms` measures the solve loop only, excluding host
  preprocessing (Ruiz scaling and power iteration), on both engines. Keep any new
  timing consistent with that.

## Reporting culture

Results in `docs/STATUS.md`, `README.md`, and `docs/writeup.md` are deliberately
honest about limits — the f32 wall that produces `IterationLimit` rows, the 2/6
infeasibility recall, the e226 sign-convention footnote. When you update a number,
**keep its caveats attached.** Never report a headline figure without its
verification basis. If a change improves a benchmark, say what hardware and what
tolerance produced it.

## Pull requests

- One logical change per PR.
- Include the test output that backs your claim. "Tests pass" without the output is
  not evidence; if you changed GPU code, that means the `--include-ignored` run.
- If you changed a documented number anywhere, update `docs/STATUS.md` in the same
  PR.
- New behavior needs a test. Bug fixes need a test that fails before the fix.

## Licensing

Sundial is dual-licensed MIT OR Apache-2.0. By contributing, you agree that your
contribution is licensed under the same terms, with no additional conditions.
