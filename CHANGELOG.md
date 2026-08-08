# Changelog

Notable changes to Sundial. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the crates follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and are
versioned together from the workspace.

Numbers here carry the same rule as the rest of the project's docs: no headline without
its verification basis. See [`docs/STATUS.md`](docs/STATUS.md) for the full record.

## [0.1.0] — unreleased

Initial release. Nothing is published to crates.io or npm yet.

### Solver

- Restarted PDHG (the PDLP family) as WebGPU compute shaders via `wgpu`, running natively
  on Metal/Vulkan/DX12 and in the browser as wasm.
- Matrix-free operator abstraction (`LinOp`) alongside the explicit sparse path, so
  structured problems never materialize a constraint matrix.
- Ruiz + Pock–Chambolle equilibration, and a CPU f64 reference implementation of the same
  algorithm for cross-checking.
- Grid-stride kernels throughout; dispatch is capped at 4,096 workgroups.
- Movement-based primal weight (`SolveOptions::movement_weight`, CLI `--movement-weight`),
  **opt-in and off by default**. On the GPU Netlib sweep it takes 20/32 → 30/32 `Optimal`
  with 5.2× fewer iterations on instances that already solved, but two newly-`Optimal`
  instances (lotfi, bnl1) land outside the ≤1e-3 objective band the published table
  guarantees. The default stays off so that claim holds; the adjudication is in
  `docs/STATUS.md`.

### Verification

- `Optimal` is only ever returned after an independent CPU-f64 KKT recheck at the exact
  returned point.
- `Infeasible` / `Unbounded` are only ever returned after a CPU-f64 Farkas certificate
  (dual ray / primal recession ray) verifies the GPU detector's nomination.

### Interfaces

- `sundial-mps` — MPS parser with no wasm or GPU dependencies.
- `sundial-cli` — `sundial solve | transport | bench | report`.
- `sundial-lp` (directory `crates/sundial-web`) — wasm-bindgen bindings and the npm package.
- Web demo: 1M-variable transport hero, drop-a-file benchmark page, and a Manhattan taxi
  matching page built on 2015 NYC TLC data (provenance in [`NOTICE`](NOTICE)).

### Measured at this release

- **Netlib sweep:** 32/32 parsed, 20/32 verified `Optimal`, 12 honest `IterationLimit`,
  0 parse errors, 0 false status claims. Every `Optimal` instance matches the published
  optimum to better than 1e-3. The `IterationLimit` rows are caused by primal-weight
  imbalance on the unweighted equilibrated path, not by f32 precision — an earlier "f32
  wall" attribution was disproved on the CPU f64 reference (`docs/STATUS.md`). e226
  carries a sign-convention footnote.
- **Infeasibility detection:** 2 of 6 certified on Netlib's infeasible/unbounded set,
  4 honest `IterationLimit`, 0 false positives.
- **Transport hero:** 1,048,576 variables verified to mu 9.83e-5 in ~9.4 s native on an
  M4 Pro, and interactively in-browser.
- **Taxi matching:** 1,024 riders × 1,152 cabs, `Optimal` in 4,288 iterations / 2,619 ms
  native.

### Known limits

- Metal forces fast-math (`fastMathEnabled=YES`, no control surface in wgpu 30), so
  error-free transforms such as df64 collapse to plain f32 there. Closed experiment; see
  [`docs/notes/df64-experiment.md`](docs/notes/df64-experiment.md).
- bore3d and kb2 fail under both weight configurations.
- GPU tests are `#[ignore]`d so CI stays green on GPU-less runners; run them locally with
  `--include-ignored`.
