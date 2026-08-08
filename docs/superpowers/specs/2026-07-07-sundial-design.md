# Sundial — Design Spec

**Date:** 2026-07-07 · **Status:** approved by user (sections 1–3 approved in brainstorming session)
**Origin:** top-ranked proposal in [`or-project-proposals.md`](../../or-project-proposals.md) (adversarially verified: zero prior art crosses WebGPU with any LP algorithm).

## Goal

The first WebGPU-native linear programming solver: restarted PDHG (the PDLP algorithm family) implemented as WGSL compute shaders via Rust + wgpu, running on any GPU (Apple/AMD/Intel/NVIDIA) both natively and in the browser with zero install. Ship: (1) a visual hero demo — optimal transport at 1M+ variables live in a browser tab, (2) a credibility benchmark page + CLI with honest numbers on standard instances, (3) an embeddable npm package.

**Non-goals:** matching CUDA data-center throughput; general presolve; MIP; f64 GPU arithmetic; beating simplex on small instances (we openly state simplex wins there).

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| Demo centerpiece | Visual hero (optimal transport) + benchmark appendix | "Race CPU on small Netlib" would *lose* (simplex wins small); OT is generated client-side, scales to GPU capacity, and convergence is visible as a picture |
| Architecture | Library-first Cargo workspace | Native CLI = cargo-speed TDD + benchmarks without browser automation; npm package falls out at M2 |
| Precision | GPU f32 iterates + compensated reductions; **reported tolerance always CPU-f64-recomputed** | WebGPU has no f64 (verified); cuPDLP-C ships f32 mode; honesty by construction |
| Headline tolerance | Relative KKT ≤ 1e-4 | PDLP's own "moderate accuracy" tier; df64 path for tighter tolerances is an M2 experiment |
| Restart scheme | Strategy trait: classic PDLP adaptive restarts first, Halpern/reflected (cuPDLPx-style) second | cuPDLPx's restarted-Halpern PDHG is still the frontier (unchanged since 2025-09); pick empirically |
| Reductions | Workgroup-shared-memory tree only | wgpu subgroups are **native-only on web** (verified 2026-07-07); portable path required |
| Iteration timing | CPU `performance.now()` per K-iteration batch (browser); timestamp queries native-CLI-only | Browser timestamp queries are quantized/gated |
| Host sparse repr | Hand-rolled CsrMatrix (M0) | Plan's Task 2 defined a minimal CSR (mul/transpose) and nothing needed more; sprs deferred until a real need appears (final-review adjudication) |
| wgpu version | 30.0.0, fallback 29.0.4 | 30.0.0 released 2026-07-01; decide in week 1 after a scaffold smoke test. Note v29 renamed push constants → "immediates" |
| License | Dual MIT / Apache-2.0 | Rust ecosystem norm |
| Repo | `git init` at `or-fable/` root | This directory is the project home |

## Architecture

```
or-fable/
├─ crates/
│  ├─ sundial-core/    # solver: problem repr, scaling, engine, WGSL kernels, termination
│  │  └─ src/shaders/*.wgsl
│  ├─ sundial-mps/     # MPS/LP parser (pure, no GPU deps, fuzzable)
│  └─ sundial-cli/     # native binary: solve/bench, JSON/CSV output
├─ web/                # Vite + TS: transport hero page + benchmark page
└─ docs/
```

### sundial-core

**Problem representation — two forms:**
1. **Explicit:** CSR matrix (minimal hand-rolled `CsrMatrix`; see decisions log) with two-sided constraint bounds `l_c ≤ Ax ≤ u_c` and variable bounds `l_v ≤ x ≤ u_v` (PDLP standard form; what MPS RANGES/BOUNDS require).
2. **Matrix-free:** the problem supplies `Ax` and `Aᵀy` as GPU kernels (a `LinOp` trait with GPU dispatch hooks) instead of a stored matrix. This is what makes the transport hero possible: its constraint operator is "row sums / column sums" and its cost matrix is computed in-shader from cell coordinates — never materialized. PDLP is matvec-based, so this is faithful to the algorithm, not a special case.

**engine:** wgpu device/queue management, buffer lifecycle, PDHG iteration loop with restart policy. Identical code paths native (Metal/Vulkan/DX12) and wasm32/WebGPU. Golden rule: *state lives on the GPU*; CPU readback only for periodic diagnostics (a handful of floats) and the final solution.

**WGSL kernel set (portable subset only — no subgroups, no f16, no f64):**
- CSR SpMV (`Ax`, `Aᵀy`; transpose stored as second CSR)
- Vector ops: axpy, scale, combine/extrapolate
- Box projection (clamp to variable/constraint bounds)
- Tree reductions: dot, ‖·‖₂, ‖·‖∞ (workgroup shared memory, two-pass)
- Compensated (two-float) accumulation inside reduction kernels
- Per-problem operator kernels (transport: row/col sums, in-shader cost)

**termination:** primal residual, dual residual, duality gap (relative, PDLP-style criteria); iteration + wall-clock caps; NaN watchdog (f32 overflow → controlled halt with honest status; auto-recovering restart is a later refinement).

**Public API:**
```rust
Solver::new(gpu_context) -> Solver
solver.solve(&problem, &opts, on_progress) -> Result<Solution>
// on_progress: FnMut(ProgressEvent { iter, primal_res, dual_res, gap, ms_per_iter })
// Solution { x, y, status: Optimal | IterLimit | TimeLimit | NumericalBreakdown, stats }
```
One interface powers CLI and web. No solver-internal types leak into consumers.

### sundial-mps
Fixed + free MPS format; OBJSENSE, RANGES (with the standard sign-convention corners), BOUNDS (MI/PL/FR/FX/BV rejected-if-integer), gzip decompression (benchmark sets ship `.gz`). Emits the explicit problem form. Line-numbered parse errors.

### sundial-cli
`sundial solve file.mps[.gz] --tol 1e-4 --json` · `sundial bench <dir> --out results.csv`. Runs the same WGSL on native backends. Timestamp-query kernel profiling behind a flag. Compares objectives against a bundled known-optima file for Netlib.

### web/
Vite + TS consuming the wasm-bindgen/wasm-pack build.
- **Transport hero:** two 2D mass distributions (presets first; draw-your-own at M2); canvas rendering of mass flow as the solve progresses; iter/gap/ms-per-iter HUD; GPU name displayed. Two 32×32 grids → 1,048,576 variables (safe within spec-guaranteed buffer limits: each full-size vector buffer ≈ 4 MiB). Two 64×64 grids → 16.7M variables ≈ 67 MiB per vector buffer — still under the 128 MiB per-binding guarantee; enabled when the probed adapter allows total memory.
- **Benchmark page:** drop `.mps/.mps.gz`, live convergence chart, results table with links to published CPU/CUDA numbers for the same instances.
- Capability detection: no-WebGPU → friendly explainer page (verified floor: Chrome/Edge 113+, Firefox 141+ Win/145+ macOS, Safari 26+; ~83–84% of traffic).

### Data flow
```
MPS file ─→ sundial-mps ─→ explicit problem ┐
transport UI ─→ generator ─→ matrix-free problem ┤→ scaling (Ruiz + Pock-Chambolle, host, once)
                                                 ┤→ GPU buffers
                                                 └→ iterate loop:
                                                     [K inner iterations, GPU-only]
                                                     → readback residual scalars
                                                     → restart decision
                                                     → ProgressEvent → UI/CLI
                                             termination → CPU f64 KKT verification → Solution
```

## Algorithm & numerics

Restarted PDHG on the LP saddle-point problem, PDLP standard form. Per iteration: one `Ax`, one `Aᵀy`, vector updates, box projections — all O(nnz)/O(n), no factorizations.

**Enhancement ladder (build in this order):**
1. Ruiz equilibration (~10 iterations) + Pock-Chambolle diagonal scaling — host-side preprocessing. Applies to explicit problems; matrix-free problems supply analytic scaling factors or opt out (the transport operator is an all-ones incidence structure — already perfectly balanced, no scaling needed).
2. **Adaptive restarts** on normalized duality-gap decay (classic PDLP policy) — the biggest convergence lever. `RestartPolicy` trait; Halpern/reflected variant added second; empirical winner ships as default.
3. Primal-weight balancing (adapt τ/σ ratio from residual balance at restarts).
4. Step size from power-iteration estimate of ‖A‖ (GPU); Malitsky-Pock adaptive steps only if fixed steps underperform.

**Precision policy:** f32 GPU iterates; compensated accumulation in reductions; CPU f64 KKT recomputation at every restart cycle and at termination — **the reported achieved-tolerance is always the CPU f64 number**. GPU-side f32 residuals drive UI display and restart decisions only.

**Transport formulation:** discrete OT between two grids (source n cells, target m cells): variables X ∈ R^(n×m) ≥ 0, row sums = source masses, col sums = target masses, cost = Euclidean (computed in-shader). Feasible by construction (masses normalized), so infeasibility handling is not needed for the hero. Rendering shows partial-flow interpolation as iterates converge.

**Correctness oracles:**
- Constructed-KKT LPs: generate (x*, y*) satisfying optimality conditions, derive problem data → exact known optimum, no external solver needed.
- Netlib known optimal objective values (bundled data file): assert relative objective error ≤ 1e-3 when solved to KKT 1e-4 (empirically safe starting bound; tighten per-instance once measured).
- CPU f64 reference PDHG (simple, slow, obviously correct) — the comparison target for every GPU kernel and for the full algorithm on tiny instances.

## Platform constraints (verified 2026-07-07)

- WebGPU has **no f64** anywhere; **subgroups are native-only in wgpu's web backend**; browser timestamp queries quantized to 100µs and flag-gated.
- Spec-guaranteed minimums: `maxStorageBufferBindingSize` = 128 MiB, `maxBufferSize` = 256 MiB. Startup probes `adapter.limits` and requests maxima; oversized problems get an explicit "needs X MiB per buffer, adapter allows Y" error. Splitting >128 MiB matrices across bindings: M2.
- WGSL→MSL via naga is mature; known portability caps (expression nesting ~256 on Metal) are far above our small kernels. WebGPU bounds-check overhead on array indexing is real but unquantified — benchmark early, keep indexing patterns provably in-bounds where possible.

## Error handling

| Failure | Behavior |
|---|---|
| No WebGPU | Capability page: what's needed, which browsers |
| Adapter limits too small | Explicit per-buffer MiB message |
| MPS parse error | Line + column + offending token |
| Non-convergence at caps | `IterLimit`/`TimeLimit` status with best residuals — never report a solution as optimal without the f64 check passing |
| NaN/Inf in iterates | `NumericalBreakdown` status; watchdog checks at restart cadence |
| Unbounded/infeasible input | M0–M1: hits iteration cap with diverging residuals, reported honestly as non-convergence; proper certificates (PDLP divergence detection) are M2 |

## Testing & CI

1. **Parser:** golden tests on real Netlib files; corner-case suite (RANGES signs, MI/FR/PL/FX bounds, fixed/free format, comments, gzip).
2. **Kernels:** every WGSL kernel vs CPU f64 reference on randomized inputs, tolerance-aware assertions.
3. **Algorithm:** CPU reference PDHG on constructed-KKT LPs (exact oracle); GPU solver on small Netlib to 1e-4 with objective assertions.
4. **CI (GitHub Actions, no GPU):** full CPU-reference suite + small GPU tests on lavapipe (Mesa software Vulkan — established wgpu CI pattern; validate in week 1, fall back to CPU-only CI + local GPU test script if lavapipe misbehaves). Metal runs locally before releases. Headless-Chrome wasm smoke test (afiro) best-effort.
5. **TDD discipline:** kernels and parser features are written test-first against the references.

## Milestones

**M0 — first weekend.** Workspace scaffold; parser subset sufficient for small Netlib instances; CPU reference PDHG solving ≥3 of them; GPU engine with the portable kernel set solving the same to 1e-4 (f64-verified); minimal web page: instance picker + live convergence chart. **Done =** `cargo test` green including Netlib assertions; page solves in-browser on the dev Mac.

**M1 — weeks 2–4.** Matrix-free `LinOp` + transport kernels; **hero demo ≥1M variables interactive on M-series Mac**; benchmark page; CLI sweeps; first honest comparison table (Netlib subset + a few large instances) vs published HiGHS/PDLP numbers.

**M2 — weeks 5–8.** npm package (name availability checked; fallback scoped `@sundial/solver`); hero polish (presets, draw-your-own); df64 experiment; infeasibility detection; GPU-presolve paper (arXiv 2604.23951) evaluated; buffer-splitting for >128 MiB matrices if needed; launch writeup with an explicit honest-limits section (f32, 1e-4 tier, no presolve, where CPU simplex wins).

**Launch bar:** hero at 1M+ vars → 1e-4 interactive on a MacBook; ≥25 Netlib + ≥5 large instances in the table; `npm install` works; writeup done.

## Data sources (verified live during proposal vetting, 2026-07-06/07)

- Netlib LP set: http://www.netlib.org/lp/data/ (~90 MPS instances + known optima)
- Mittelmann benchmark instances & published solver times: https://plato.asu.edu/bench.html (updated 2026-07-01)

## Kill risks (carried from proposal, with mitigations)

1. **f32 numerics wall** — PDHG may stall pre-1e-4 on ill-conditioned instances. Mitigations: scaling (step 1 of ladder), restarts, compensated reductions; honest per-instance reporting of achieved tolerance; df64 path in reserve.
2. **WebGPU ceilings/jank** — per-adapter limits and browser quirks shrink headline claims. Mitigations: probe-and-report design, hero sized within spec-guaranteed limits, cross-browser testing from M0.
3. **Utility skepticism** — "why solve LPs in a browser?" Mitigations: npm-embeddable library as a first-class artifact (private data never leaves the client; no backend needed), plus the native CLI showing this is a real solver, not a stunt.
