# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Sundial: a linear-programming solver (restarted PDHG, the PDLP family) that runs as WebGPU compute shaders via wgpu — natively (Metal/Vulkan/DX12) and in the browser as wasm. Current state and milestone history live in `docs/STATUS.md`; read it first when picking up work. Design documents are under `docs/design/`. If a working copy contains `.superpowers/sdd/`, `docs/superpowers/`, or `docs/or-project-proposals.md`, those are gitignored local planning scratch and are not part of the repository — never reference them from published docs or code comments.

## Commands

```bash
# Tests
cargo test --workspace                        # CPU suite (what CI runs)
cargo test --workspace -- --include-ignored   # + GPU suite (needs a GPU); run the 1M transport gate in --release
cargo test -p sundial-core --test kkt         # one integration-test file
cargo test -p sundial-core --test gpu_transport -- --include-ignored --nocapture  # one GPU test file

# Lint gates (CI fails on any of these)
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build -p sundial-lp --target wasm32-unknown-unknown
cargo clippy -p sundial-lp --target wasm32-unknown-unknown -- -D warnings

# CLI
cargo run -p sundial-cli --release -- solve crates/sundial-mps/tests/fixtures/afiro.mps
cargo run -p sundial-cli --release -- transport --grid 32     # 1M-variable hero (~9.4 s on M4 Pro)
cargo run -p sundial-cli --release -- bench <dir> --out results.csv
cargo run -p sundial-cli --release -- report results.csv --out report.md

# Web demo (Vite + TS; pages import from crates/sundial-web/pkg/)
wasm-pack build crates/sundial-web --target web   # or: bash scripts/build_npm.sh (adds npm packaging steps)
cd web && npm install && npm run dev              # index.html = transport hero, bench.html = drop-a-file
cd web && npx tsc --noEmit && npm run build

# Full fresh-clone verification (all of the above from a scratch clone)
bash scripts/verify_clean_checkout.sh

# Netlib instances for benching (bench/ is gitignored, reproducible)
bash scripts/fetch_netlib.sh
```

GPU tests are `#[ignore]`d so CI (GitHub runners, no GPU) stays green; always run them locally with `--include-ignored` before claiming a GPU change works.

**Never run `npm publish`.** The `sundial-lp` package is deliberately packed-but-unpublished; publishing is a human-run step on `RELEASE.md`, as is everything else on that checklist.

## Architecture

Workspace of four crates:

- `crates/sundial-core` — everything solver. `problem.rs` (types, `SolveOptions`, `Solution`), `scale.rs` (Ruiz + Pock–Chambolle equilibration), `reference.rs` (CPU f64 reference PDHG), `kkt.rs` (CPU f64 KKT verification), `farkas.rs` (CPU f64 infeasibility/unboundedness certificates), `linop.rs` (matrix-free operator abstraction), `transport.rs` (optimal-transport instance generation, presets, draw-your-own masses), `weight.rs` (PDLP primal-weight ω), `gpu/` (wgpu engine, buffers, kernel dispatch; WGSL shaders in `gpu/shaders/`).
- `crates/sundial-mps` — MPS parser, pure (no wasm/GPU deps).
- `crates/sundial-web` — wasm-bindgen bindings; **the crate/package is named `sundial-lp`** (so `cargo build -p sundial-lp` but the directory is `sundial-web`). Wasm-only via `#![cfg(target_arch = "wasm32")]`; compiles as an empty rlib on native so `cargo test --workspace` passes. `types-extra.d.ts` is hand-maintained (generated bindings type serde results as `any`) and must ship in the npm tarball.
- `crates/sundial-cli` — `sundial solve | transport | bench | report`.

`web/` is a separate Vite+TS app (not a crate) with two entry pages; it consumes the local wasm-pack output at `crates/sundial-web/pkg/sundial_lp`.

### Invariants — do not weaken these

- **Certificate honesty.** `Optimal` is only ever set after an independent CPU-f64 KKT recheck at the exact returned point (dual sign-projected first). `Infeasible`/`Unbounded` are only ever set after a CPU-f64 Farkas certificate (dual ray / primal recession ray) verifies the GPU streak-detector's nomination; a failed verification resets the streak to 0. The GPU never grades its own homework. A missed detection (honest `IterationLimit`) is acceptable; a false status claim is not — don't "fix" low recall by relaxing verification.
- **GPU infinity convention:** ±1e30 sentinels in every GPU buffer; shaders never do inf arithmetic (browser fast-math portability).
- **Grid-stride kernels everywhere** — dispatch is capped at 4,096 workgroups; correctness beyond that comes from stride loops, not bigger dispatches.
- **Metal forces fast-math** (wgpu 30 hardcodes `fastMathEnabled=YES`, no control surface). Error-free transforms (df64, Neumaier compensation) silently collapse to plain f32 on Metal — don't build precision features on them; see `docs/notes/df64-experiment.md` before touching accumulation code.
- **Primal-weight ω applies only to the matrix-free/unscaled path.** The explicit Ruiz+PC-equilibrated path runs at τ=σ (ω limit-cycles there — M1 Task 6a adjudication).
- **Timing convention:** `solve_ms` measures the solve loop only, excluding host preprocessing (Ruiz scaling + power iteration), on both engines.

### Reporting culture

Results in `docs/STATUS.md`, `README.md`, and `docs/writeup.md` are deliberately honest about limits (12 `IterationLimit` rows in the Netlib sweep, 2/6 infeasibility recall, e226 sign-convention footnote). When updating numbers, keep the caveats attached; never report a headline without its verification basis. Note the "f32 wall" attribution for those `IterationLimit` rows was disproved on 2026-08-08 — don't reintroduce it; see `docs/STATUS.md`.
