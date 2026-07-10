# Sundial — Project Status

Updated: 2026-07-07 (M1 completion)

## Milestones

- [x] **M0** — GPU solver core (restarted PDHG in WGSL), MPS parser, CPU f64 reference, CLI, Netlib gate, minimal web demo. **Merged to main 2026-07-07.**
- [x] **M1** — matrix-free `LinOp`; optimal-transport hero at 1,048,576 variables, native + in-browser; drop-a-file benchmark page; CLI Netlib sweep tooling producing the comparison table. **All 13 implementation tasks (1–11 + 6a) complete and review-approved.**
- [ ] **M2** (spec: weeks 5–8) — npm package (per user decision 2026-07-07, this goes at the very END of the M2 plan); hero polish (presets + draw-your-own); double-double precision experiment; infeasibility detection; launch writeup ("Show HN"). Backlog: `docs/superpowers/m2-backlog.md`.

**Launch bar** (from spec): hero ≥1M vars → 1e-4 interactive on a MacBook; ≥25 Netlib + ≥5 large instances in the table; `npm install` works; writeup done.

## M1 results (verified on Apple M4 Pro / Metal, 2026-07-07)

- **1M-variable optimal-transport hero:** two 32×32 grids (blobs preset), n = 1,048,576 vars, m = 2,048 equality rows — **Optimal, 16,000 iterations, 9 restarts, ~9.4 s wall, verified mu 9.83e-5** (CPU-f64 KKT at the returned point). Reproduce: `cargo run -p sundial-cli --release -- transport --grid 32`. Gate test: `gpu_transport_1m_variables_to_1e4` (crates/sundial-core/tests/gpu_transport.rs). CPU sanity at grid 8: Optimal, 1,792 iters, 25 ms, mu 9.89e-5.
  - History, recorded honestly: the gate first failed at τ=σ (IterationLimit, 500,032 iters, 290 s, mu 1.23e-4 — dual ~1e-9, primal plateau ~1e-4). Fixed by primal-weight balancing (Task 6a): PDLP ω₀ = ‖c‖/‖q‖ (iterate space) + √-damped residual-balance update at restarts, ω ∈ [1e-4, 1e4]. **Adjudicated:** ω applies to the matrix-free/unscaled path only — on the Ruiz+PC-equilibrated explicit path it limit-cycled (0.02↔0.12) and regressed share2b; the explicit path is proven bit-identical to the pre-ω τ=σ behavior. Movement-based ω for the explicit path is M2 backlog.
- **Readback batching** (Task 1, M0-review prerequisite): one readback per residual check (was ~11). afiro CLI wall: ~630 ms vs 760 ms M0 record (~17% faster; readback still dominates at this tiny scale).
- **Netlib sweep** (Task 10): 32 instances fetched from netlib.org (`scripts/fetch_netlib.sh` compiles netlib's emps.c; `bench/` gitignored, reproducible). **19/32 Optimal at CPU-f64-verified 1e-4; 12 IterationLimit (the documented f32 wall — honest rows, cap 2M iters); 1 parse error** (blend.mps line 355, set-name-less RHS format — M2 backlog). Optimal includes afiro, adlittle, bandm, beaconfd, brandy, e226, etamacro, israel, recipe, sc105, sc50a, sc50b, scagr25, scagr7, stocfor1 (+ others in results.csv). IterationLimit includes agg/agg2/agg3, bnl1, bore3d, capri, kb2, lotfi, sc205, ship04s (near-miss: primal 1.551e-4). Tooling: `sundial bench <dir> --out results.csv` (name field quoted, error rows carry full anyhow chains), `sundial report results.csv --out report.md`.
  - **e226 footnote:** its netlib-readme "known optimum" uses the opposite sign convention for the objective-row RHS constant (delta = 2× the constant ≈ 7.11). Our verified optimum is −11.635074 (KKT-certified at 1e-4); the readme lists ≈ −18.75. The report shows rel err 3.6e-1 against the readme value — that's a convention mismatch, not a solver defect. The optima file is not altered; this is a footnote only.
- **Browser:** wasm API adds transportPreview / solveTransport (live marginal snapshots) / solveMpsBytes (gzip) + solveMps. Hero page (`index.html`): preset (blobs, ring→square) + grid (16×16 = 65,536 vars / 32×32 = 1,048,576 vars) pickers, three live heatmaps (source / mass arriving / target), rAF-throttled convergence chart. Bench page (`bench.html`): fixture picker + drag-drop `.mps`/`.mps.gz` + honest results table. Machine gates all pass (tsc clean, vite build emits both pages, wasm-pack build clean).
  - **Interactive browser verification: pending user confirmation** (same M0 precedent carried forward — dev servers get SIGTERM'd in this execution environment; the user verifies from their own terminal via `cd web && npm run dev`). This also closes M0's carried-over open item of the same kind, now re-opened for M1's hero page specifically.
- **Timing convention** (user-adjudicated 2026-07-07): as of M1, `solve_ms` measures the solve loop only, EXCLUDING host preprocessing (Ruiz scaling + power-iteration norm) — consistent across CPU and GPU engines. M0's numbers included preprocessing; account for this when comparing M0 vs M1 timings directly.
- Matrix-free transport operator: no constraint matrix materialized anywhere (16-byte uniform + caller buffers); the cost vector is materialized (4 MiB at g=32); in-shader cost generation (needed at g=64 / 16.7M vars, where the cost vector would be 67 MiB) is M2.
- Grid-stride kernels everywhere (dispatch cap 4,096 workgroups; correctness beyond 1,048,576 elements via stride loops).
- Certificate honesty unchanged: `Optimal` only after CPU f64 KKT at the returned point; GPU never grades its own homework.
- Suites (Task 11 full run, all green): workspace CPU suites; GPU suites incl. the 1M gate (`--include-ignored`, release for the gate); Netlib CPU 3/3 + GPU 5/5; fmt; clippy `-D warnings` (native + wasm32); wasm32 build; wasm-pack; tsc; vite build (both pages).

## M0 results (verified on Apple M4 Pro / Metal, 2026-07-07)

- **GPU Netlib gate: 5/5** fixtures (afiro, sc50a, sc50b, adlittle, share2b) reach `Optimal` at relative KKT ≤ 1e-4, CPU-f64-verified, with relative objective error ≤ 1e-3 vs published optima (worst case 6.7e-4).
- **CPU reference gate (CI): 3/3** (afiro, sc50a, sc50b) at the same tolerances.
- Example run: afiro on GPU — objective −464.7530946 (published −464.75314286), 4,352 iterations, 5 restarts, ~760 ms wall including per-check readback overhead (readback dominates at this tiny scale; the GPU pays off at M1 sizes). *(Timing convention note: this M0 number included host preprocessing; M1's `solve_ms` excludes it — see above.)*
- Suites: 18 CPU tests + 8 GPU tests (`#[ignore]`d in CI, run locally with `--include-ignored`); fmt / clippy (native + wasm32) / wasm32 build gates all green.

## Design decisions of record (full detail in spec + plan)

- **Certificate honesty:** `Optimal` is only ever set after a CPU f64 KKT recheck at the exact returned point; the dual is sign-projected first and the returned solution *is* the verified point.
- **Projected-multiplier dual objective on both row and column sides** — the duality gap stays finite and is genuinely enforced (an f64 ±inf would have turned it into NaN and silently dropped it from the certificate on every standard-form LP). Adjudicated mid-M0; plan commits 91608e5 / fcec2c0.
- **GPU infinity convention:** ±1e30 sentinels in every GPU buffer; shaders never perform inf arithmetic (browser fast-math portability).
- **Primal-weight balancing scope (M1, Task 6a):** ω (PDLP primal-weight update) applies only to the matrix-free/unscaled path; the explicit Ruiz+PC-equilibrated path keeps its original τ=σ behavior (proven bit-identical) because the residual-ratio ω update limit-cycles there. See M1 results above.
- **Timing convention (M1):** `solve_ms` excludes host preprocessing as of M1; see above.

## Key documents

- `docs/superpowers/specs/2026-07-07-sundial-design.md` — approved spec: architecture, milestones, launch bar, platform constraints
- `docs/superpowers/plans/2026-07-07-sundial-m0.md` — M0 plan (normative, includes all mid-flight adjudications)
- `docs/superpowers/plans/2026-07-07-sundial-m1.md` — M1 plan (12 tasks + 6a; all complete, includes all mid-flight adjudications)
- `docs/superpowers/m2-backlog.md` — minor items carried from M1's final review, for M2 triage
- `or-project-proposals.md` — the original OR project survey (Sundial plus 4 other adversarially-vetted proposals)

## Known gaps / notes

- No GitHub remote yet — the CI workflow (`.github/workflows/ci.yml`) has never executed on GitHub runners.
- Interactive browser verification of the M1 transport hero page: pending user confirmation (dev server runs via `cd web && npm run dev`); see M1 results above.
- Task-level execution records (briefs, implementer reports, review verdicts) were session scratch (`.superpowers/sdd/`, gitignored); their outcomes are summarized here and in the plan's adjudication commits.
