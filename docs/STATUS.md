# Sundial — Project Status

Updated: 2026-08-08 (M2 closed 2026-07-14; taxi demo merged 2026-07-19; repo pushed + CI green, still private and unpublished)

## Milestones

- [x] **M0** — GPU solver core (restarted PDHG in WGSL), MPS parser, CPU f64 reference, CLI, Netlib gate, minimal web demo. **Merged to main 2026-07-07.**
- [x] **M1** — matrix-free `LinOp`; optimal-transport hero at 1,048,576 variables, native + in-browser; drop-a-file benchmark page; CLI Netlib sweep tooling producing the comparison table. **All 13 implementation tasks (1–11 + 6a) complete and review-approved.**
- [x] **M2** (design: `docs/design/infeasibility-detection.md`) — parser hardening, infeasibility/unboundedness detection, df64 precision experiment, three new transport presets + draw-your-own masses, GPU-presolve literature memo, launch writeup + RELEASE checklist, and the `sundial-lp` npm package. **All 12 implementation tasks complete and review-approved; browser human gate passed 2026-07-14 — M2 closed.** Backlog carried forward into the M3 seeds below.

**Launch bar** (from spec): hero ≥1M vars → 1e-4 interactive on a MacBook; ≥25 Netlib + ≥5 large instances in the table; `npm install` works; writeup done. **Met** — sweep table now covers 32/32 netlib instances (20 Optimal + 12 honest IterationLimit, 0 parse errors), all of them small/medium classic-Netlib; the ≥5-large-instance bar is met by the 1M-variable transport hero, reported separately (M1/M2 results below, `docs/writeup.md`), not by additional rows in that table; `sundial-lp` packs cleanly via `npm pack` (never published — see M2 results); `docs/writeup.md` is a complete Show HN draft with `<DEMO_URL>` as its sole unfilled placeholder.

## M3 (in progress)

### 2026-08-08 — The "f32 wall" was misattributed; movement-based ω recovers 10 of the 12

**Diagnostic (decisive).** The 12 `IterationLimit` rows have been described since
M1 as "the documented f32 wall." That attribution is **wrong**. Re-solving all 12
on the **CPU f64 reference** (`--engine cpu`, 500k iters) reproduces every failure
— in f64, where no 1e-4 precision floor exists. The signature is not precision
loss but extreme step imbalance: one side collapses to machine epsilon while the
other stalls orders of magnitude above tolerance (agg: rel_primal **1.6e-16** vs
rel_dual 1.3e-2; sc205: **8.4e-17** vs rel_gap **7.7e-1**), and the direction is
*inverted* across instances (scorpion/ship04s stall on the primal with the dual
at ~1e-15). No single fixed τ=σ can serve both — which is exactly the case for an
adaptive primal weight. Root cause: per the M1 Task 6a adjudication the explicit
(Ruiz+PC) path runs **completely unweighted**.

**Fix (experiment, opt-in).** `weight::update_primal_weight_movement` implements
PDLP's movement rule ω ← ω^(1−θ)·(Δy/Δx)^θ, θ=0.5, measured between consecutive
restart points in iterate space. The structural difference from the M1
residual-ratio rule is why it works where that one failed: the residual rule
multiplies ω by the full damped ratio every restart and has **no fixed point**
unless residuals balance exactly (hence the observed 0.02↔0.12 orbit); the
movement rule is a **contraction in log space** toward log(Δy/Δx), so error halves
per restart. Pinned by `movement_update_contracts_toward_the_movement_ratio`.

**Measured (CPU f64, 500k iters, all 32 Netlib instances):** **19/32 → 28/32
Optimal — 10 wins, 1 regression.** Wins: agg, agg2, agg3, beaconfd, kb2, lotfi,
sc205, scorpion, share1b, ship04s.

**Caveats, attached:**
- **share2b is a real regression, not slow convergence.** Baseline solves it in
  93,376 iters; with movement-ω it fails at 2M with the gap *worse* than at 500k
  (1.68e-2 vs 4.84e-3). It is the **same instance** that regressed under the M1
  residual-ratio ω — share2b appears pathological for any ω adaptation on the
  equilibrated path. Unexplained; this is why the flag is opt-in.
- The 19/32 baseline above is CPU@500k and is *not* comparable to the published
  GPU@2M row (beaconfd, for one, is Optimal in the published table and only
  fails the CPU@500k baseline on the cap). GPU numbers are below.
- Default is **off** (`SolveOptions::movement_weight`, CLI `--movement-weight`);
  with it off, results are bit-identical to published behavior.
- Ratio direction was adjudicated **empirically**, not assumed: PDLP's published
  ω → Δy/Δx scored 28/32; the inverted ratio scored **3/32** (0 wins, 16
  regressions). The losing variant was deleted.

### 2026-08-08 — Movement ω ported to the GPU engine

New surface: a `reduce_diff_sq` WGSL entry (Σ(a−b)² with the same Neumaier
compensation as `reduce_dot`; differencing happens **inside** the kernel because
at a restart the two iterates are neighbours and an f32 subtract-then-dot would
round the movement away), `Reducer::record_diff_sq`/`diff_sq`, and `x_prev`/
`y_prev` restart-point buffers allocated **only when armed** — the 1M hero is
matrix-free, so it never pays for them. Per restart the engine records both
Δ-reductions into `results[12..14]`, reads them back, and rewrites τ/σ into the
two static uniforms in place, exactly as the residual-ratio path already does.
Gate is the complement of `primal_weight`, mirroring the CPU
`scaling.is_some()`. Restart cadence only, so the extra readback is ~16 per
solve against tens of thousands of iterations.

**Measured (GPU, 2M cap — the published sweep's exact settings):**
**20/32 → 30/32 Optimal, 10 wins, ZERO status regressions.** Among the 20
instances Optimal in both, **18 got faster**; total iterations across them fall
**3,451,008 → 659,264 (5.2×)**. Extremes: beaconfd 1,680,896 → 10,432 iters
(161×), agg2 2M-stall → 7,680.

**The share2b CPU regression does not reproduce on the GPU** — it stays Optimal,
though 3.3× slower (123,136 → 400,896 iters).

**Caveat that the status column hides — read before flipping the default.**
Status counts improve, but *objective accuracy against the Netlib published
optima does not uniformly improve*, and for two instances it is materially worse
than anything in the current table:

| instance | published | with movement ω |
|---|---|---|
| lotfi | IterationLimit (rel err 8.8e+0) | **Optimal, rel err 5.8e-2** |
| bnl1 | IterationLimit (rel err 2.1e-3) | **Optimal, rel err 1.2e-2** |
| agg3 | IterationLimit (rel err 5.5e-3) | Optimal, rel err 1.2e-3 |
| etamacro | Optimal, rel err 2.4e-4 | Optimal, **rel err 1.5e-3** |

All four are honest `Optimal`: each passed the independent CPU-f64 KKT recheck at
its returned point. But a relative KKT residual ≤ 1e-4 does not tightly bound
objective error on ill-conditioned or degenerate instances, and lotfi/bnl1 land
far outside the ≤1e-3 band every currently-Optimal instance sits in (worst real
case today: adlittle 6.7e-4). **Flipping the default would therefore invalidate
the README/writeup claim that every Optimal instance matches the published
optimum to better than 1e-3.** etamacro also gets *less* accurate despite already
being Optimal.

Two instances still fail (bore3d, kb2), and both degrade in residual quality even
though the status label is unchanged: bore3d's primal goes 2.4e-2 → 1.7e+2, kb2's
dual 2.8e-5 → 6.9e-2. kb2 is a *win* on the f64 reference but a degradation on
the f32 GPU — a genuine engine difference, unexplained.

**Adjudicated 2026-08-08 (user):** keep the default **off**; cite 30/32 in the
writeup as an *opt-in* result with the accuracy caveat stated inline. Rationale:
this buys the stronger number in the narrative without weakening the published
table, and it avoids trading measurable objective accuracy for an advertisable
status count — the trade this project refuses to make silently. With the default
off, every published number stands unchanged and the ≤1e-3 accuracy claim on the
Optimal rows remains true.

**Still open, lower priority:** bore3d and kb2 fail under both configurations,
and both *degrade* in residual quality under movement ω behind an unchanged
status label (bore3d primal 2.4e-2 → 1.7e+2; kb2 dual 2.8e-5 → 6.9e-2). kb2 is a
win on the f64 reference and a degradation on the f32 GPU — an engine split with
no explanation yet. Worth a look before any future default flip, since a flip
would ship that degradation silently.

### 2026-07-19 — Taxi demo (M3 seed): Manhattan matching page

`web/taxi.html`: 1,024 real TLC riders × 1,152 cabs (2015-06 slice, the last
public data with exact GPS) solved as a capacitated matching over the existing
matrix-free `TransportOp` — new surface is only `problem_from_points`, the
`recover` module (integral matching recovery + certified dual floor), and the
`solveMatching` wasm entry (`dominant_assignment` is test-only, superseded by
recovery); engine/shaders untouched. Native gate (`gpu_matching`, --release):
Optimal in 4,288 iters / 2,619 ms; recovered plan 8.676982 units vs certified
floor 8.647033 (slack 1968.9 ft, ~1.9 ft/pickup). Browser (Chrome/Metal, M4
Pro): ~2.2–2.5 s per solve; greedy nearest-neighbor dispatch measured 25%
worse than optimal on the fixture. Honesty: "proven optimal" copy is gated
on CPU-f64 verified status AND certified-floor slack ≤ 5e-3 of total (the
recovery contract — the dual-repair floor, not a dominance readout); free-cab
positions are a drop-off proxy (disclosed on-page); distances are
straight-line.
loop.mp4/poster.png for the no-WebGPU card are a pending human capture step.

## M2 results (verified on Apple M4 Pro / Metal, 2026-07-10)

- **Netlib sweep refresh** (Tasks 1–2, 11): parser now accepts set-name-less RHS lines (real-world netlib corner) and the `up_negative` flag resets correctly on repeated `UP` lines for the same column. blend.mps, previously a parse error, now solves **Optimal (−30.8119660669, 7,488 iters, 1.1 s)** against optima-file value −30.812149846 (gap within verified mu 9.98e-5). Full sweep: **20/32 Optimal, 12 IterationLimit (honest non-solves; the "f32 wall" attribution carried here was disproved 2026-08-08 — see M3 above), 0 parse errors** (was 19/32 + 1 parse error at M1). e226's sign-convention footnote (docs/writeup.md, README) still applies and is unchanged by this refresh.
- **Infeasibility / unboundedness detection** (Tasks 3–4): CPU Farkas-certificate verification plus a GPU-side streak detector (monotonic-with-margin growth threshold GROWTH=1.02, chosen after proving PDHG divergence on an infeasible/unbounded instance is *linear*, not geometric, so a looser geometric streak could structurally never trigger — Applegate et al.). Verification failure resets the streak to 0 (matches CPU semantics; delays rather than misses detection under sustained divergence). **Field data on the netlib infeasible/unbounded set (6 instances): 2/6 certified Infeasible** (itest2, galenet, both @ 12,544 iters), **4 honest IterationLimit, 0 false Optimal** — the certificate path is structurally incapable of a false positive (an Optimal claim always requires the independent CPU-f64 KKT recheck; a streak claim always requires a Farkas certificate check), so the 2/6 recall is an honest, if partial, detection rate rather than a tuning shortfall. Both constructed Farkas oracles (primal-infeasible, dual-unbounded) certify at iteration 12,352 (3 restarts) under GROWTH=1.02, hand-traced against the math by the reviewer.
- **df64 (double-double) precision experiment** (Tasks 5–6): **DEFERRED — mechanism-level, not a tuning outcome.** Source-verified finding: on wgpu 30 / Metal, `MTLCompileOptions.fastMathEnabled=YES` is hardcoded with no wgpu/naga control surface (checked wgpu-hal `metal/device.rs`, naga `msl::Options`), so Metal's fast-math folds `fma(a,b,-p)` to `0`, destroying the error-free-transform arithmetic df64 depends on — df64 compiles and runs correctly (machinery is sound; `--df64` solves still reach `Optimal`) but is byte-identical to plain f32 on this backend. **Bonus finding, no gate impact:** the same fast-math collapse silently degrades the *existing* M0 Neumaier compensated-summation kernel (`reduce.wgsl`) to plain f32 as well (e32==e64 exactly) — the 1e-4 verified tier is unaffected, but the "compensated accumulation" doc claim needs a Metal caveat (now carried into the honest-limits section of docs/writeup.md). Full analysis: `docs/notes/df64-experiment.md`. Revisit condition below (M3 seeds).
- **Hero polish** (Tasks 7–8): three new transport presets (spiral, checker, corners) alongside blobs and ring→square; **draw-your-own masses** mode lets a user paint source/target mass by hand in the browser and solve the resulting custom optimal-transport instance (junk-value cleaning + per-side normalization in `transport::problem_from_masses`).
- **GPU-presolve literature memo** (Task 9, timeboxed): read Cederberg & Boyd (arXiv 2604.23951, PSLP). **Recommendation: DEFER.** The technique applies only to the explicit-CSR path (never the matrix-free transport path); the paper's own benchmarks show wins concentrated in a minority of large/slow instances with a ~52–59% win rate elsewhere (net-negative overhead on problems already solved quickly); and postsolve correctness engineering — mapping every reduction's inverse back to the original problem before Sundial's CPU-f64 KKT re-verification — is a multi-week, certificate-honesty-critical effort. Full memo: `docs/notes/gpu-presolve-memo.md`. Revisit condition below.
- **npm package `sundial-lp`** (Task 12, final): wasm bindings migrated out of `sundial-mps` into a dedicated publishable crate, `crates/sundial-web` (package name `sundial-lp`). **Name-check: `npm view sundial-lp` → `npm error code E404` / `404 Not Found` — available**, no fallback name needed. Built via `scripts/build_npm.sh` (`wasm-pack build --release` + `npm pack --dry-run`); `npm pack` tarball (`sundial-lp-0.1.0.tgz`, 180.9 kB) contains exactly the expected 8 files — `LICENSE-APACHE`, `LICENSE-MIT`, `README.md`, `package.json`, `sundial_lp_bg.wasm`, `sundial_lp.d.ts`, `sundial_lp.js`, `types-extra.d.ts` (hand-maintained result-shape types, since generated bindings type `serde_wasm_bindgen` results as `any`). Also added a `webgpuAvailable()` capability probe (checks `navigator.gpu` without requesting a device). **`npm publish` was never run — packed, not published**, per the milestone decision. `web/` (both demo pages) repointed to `crates/sundial-web/pkg/sundial_lp`; CI and `scripts/verify_clean_checkout.sh` repointed to build/lint `-p sundial-lp` / `crates/sundial-web`. One packaging gap found and fixed during implementation: wasm-pack copies `LICENSE-APACHE`/`LICENSE-MIT` into `pkg/` but does not add them to `package.json`'s `files` allowlist (only a bare `LICENSE`/`LICENCE` is auto-packed by npm), so `build_npm.sh` now explicitly lists both via `npm pkg set`.
- **Launch artifacts** (Task 11): `docs/writeup.md` — a complete 9-section Show HN draft with `<DEMO_URL>` as its sole placeholder; `RELEASE.md` — the human-run publish checklist (repo-public → CI → demo deploy → npm publish → post), nothing on it executed. Both already anticipate the `sundial-lp` / `crates/sundial-web` naming finalized in this task.
- Suites (Task 12 final run, all green): workspace CPU suites incl. `sundial-lp` (0 tests, empty crate off wasm32 by design); GPU suites incl. the 1M transport gate and both Netlib gates (`--include-ignored`, CPU 3/3 + GPU 5/5); fmt; clippy `-D warnings` (native workspace + wasm32 `-p sundial-lp`); `npm pack` tarball inspected; tsc clean; vite build (both dist pages, wasm asset now `sundial_lp_bg-*.wasm`); `scripts/verify_clean_checkout.sh` (fresh clone, full rebuild) green.
- **Browser human gate: PASSED, user-confirmed 2026-07-14** (wasm-pack rebuild + `cd web && npm run dev`, same session/dev-server pattern as M0/M1): a new transport preset solved to `Optimal (CPU f64 verified)` on the hero page; a draw-your-own masses instance solved to `Optimal`; an infeasible-set `.mps` drop on the bench page reported certified `Infeasible` — no false `Optimal`, no error row. This was the last open M2 closure item.

## M1 results (verified on Apple M4 Pro / Metal, 2026-07-07)

- **1M-variable optimal-transport hero:** two 32×32 grids (blobs preset), n = 1,048,576 vars, m = 2,048 equality rows — **Optimal, 16,000 iterations, 9 restarts, ~9.4 s wall, verified mu 9.83e-5** (CPU-f64 KKT at the returned point). Reproduce: `cargo run -p sundial-cli --release -- transport --grid 32`. Gate test: `gpu_transport_1m_variables_to_1e4` (crates/sundial-core/tests/gpu_transport.rs). CPU sanity at grid 8: Optimal, 1,792 iters, 25 ms, mu 9.89e-5.
  - History, recorded honestly: the gate first failed at τ=σ (IterationLimit, 500,032 iters, 290 s, mu 1.23e-4 — dual ~1e-9, primal plateau ~1e-4). Fixed by primal-weight balancing (Task 6a): PDLP ω₀ = ‖c‖/‖q‖ (iterate space) + √-damped residual-balance update at restarts, ω ∈ [1e-4, 1e4]. **Adjudicated:** ω applies to the matrix-free/unscaled path only — on the Ruiz+PC-equilibrated explicit path it limit-cycled (0.02↔0.12) and regressed share2b; the explicit path is proven bit-identical to the pre-ω τ=σ behavior. Movement-based ω for the explicit path is M2 backlog.
- **Readback batching** (Task 1, M0-review prerequisite): one readback per residual check (was ~11). afiro CLI wall: ~630 ms vs 760 ms M0 record (~17% faster; readback still dominates at this tiny scale).
- **Netlib sweep** (Task 10): 32 instances fetched from netlib.org (`scripts/fetch_netlib.sh` compiles netlib's emps.c; `bench/` gitignored, reproducible). **19/32 Optimal at CPU-f64-verified 1e-4; 12 IterationLimit (honest rows, cap 2M iters — originally attributed to an f32 precision wall; that attribution was disproved 2026-08-08, see M3 above); 1 parse error** (blend.mps line 355, set-name-less RHS format — M2 backlog). Optimal includes afiro, adlittle, bandm, beaconfd, brandy, e226, etamacro, israel, recipe, sc105, sc50a, sc50b, scagr25, scagr7, stocfor1 (+ others in results.csv). IterationLimit includes agg/agg2/agg3, bnl1, bore3d, capri, kb2, lotfi, sc205, ship04s (near-miss: primal 1.551e-4). Tooling: `sundial bench <dir> --out results.csv` (name field quoted, error rows carry full anyhow chains), `sundial report results.csv --out report.md`.
  - **e226 footnote:** its netlib-readme "known optimum" uses the opposite sign convention for the objective-row RHS constant (delta = 2× the constant ≈ 7.11). Our verified optimum is −11.635074 (KKT-certified at 1e-4); the readme lists ≈ −18.75. The report shows rel err 3.6e-1 against the readme value — that's a convention mismatch, not a solver defect. The optima file is not altered; this is a footnote only.
- **Browser:** wasm API adds transportPreview / solveTransport (live marginal snapshots) / solveMpsBytes (gzip) + solveMps. Hero page (`index.html`): preset (blobs, ring→square) + grid (16×16 = 65,536 vars / 32×32 = 1,048,576 vars) pickers, three live heatmaps (source / mass arriving / target), rAF-throttled convergence chart. Bench page (`bench.html`): fixture picker + drag-drop `.mps`/`.mps.gz` + honest results table. Machine gates all pass (tsc clean, vite build emits both pages, wasm-pack build clean).
  - **Interactive browser verification: CONFIRMED by user 2026-07-10** (screenshots): hero at 32×32 (1,048,576 vars) reaches `Optimal (CPU f64 verified)` in-browser — 16,000 iterations at 0.547 ms/iter (≈9 s solve) on `apple (BrowserWebGpu)`, arriving-mass panel visually converged to the target; 16×16 in 1,000 ms wall (8,256 iters, verified residuals ≤ 4.8e-5); bench page solves afiro to Optimal (−464.75309, 380 ms wall). This also closes M0's carried-over browser-verification item.
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
- **Infeasibility-detection streak semantics (M2, Task 3):** divergence growth threshold GROWTH=1.02 (monotonic-with-margin, not geometric) — proven necessary because PDHG divergence on infeasible/unbounded instances is linear, so a geometric streak could structurally never fire. A failed verification resets the streak to 0. `Infeasible`/`Unbounded` status is only ever set after an independent Farkas-certificate check, mirroring `Optimal`'s CPU-f64-KKT-recheck rule — both false-positive paths are structurally closed off. See M2 results above.
- **df64 deferred at the mechanism level (M2, Task 5–6):** not a measured-and-rejected precision tier — Metal's forced fast-math makes df64 provably byte-identical to f32 on the current backend, so no sweep could show a difference. Revisit condition in M3 seeds below.
- **GPU presolve deferred (M2, Task 9):** applicable only to the explicit-CSR path, with real overhead risk and non-trivial postsolve-correctness engineering; revisit condition in M3 seeds below.
- **npm packaging (M2, Task 12):** `sundial-lp` publishable crate is wasm-only by construction (`#![cfg(target_arch = "wasm32")]` at the crate root) — it still satisfies `cargo test --workspace` on native targets by compiling as a (trivially empty) rlib, so the wasm-bindgen/js-sys dependency set did not need target-gating in Cargo.toml.

## Key documents

- `docs/design/architecture.md` — architecture, algorithm and numerics, platform constraints
- `docs/design/infeasibility-detection.md` — Farkas certificate math behind `farkas.rs`; df64 and npm packaging design
- `docs/design/taxi-demo.md` — design of the Manhattan matching page
- `docs/notes/df64-experiment.md` — full df64 findings (Metal fast-math source trace, Neumaier collateral, decision gate)
- `docs/notes/gpu-presolve-memo.md` — GPU-presolve literature memo (Cederberg & Boyd, arXiv 2604.23951) and defer rationale
- `docs/writeup.md` — Show HN launch draft (`<DEMO_URL>` is the sole placeholder)
- `RELEASE.md` — human-run publish checklist (repo public → CI → demo deploy → npm publish → post); §1 (repo created, pushed) and §2 (CI green) done, everything from §3 (demo deploy) on still unexecuted

## Known gaps / notes

- Remote is `https://github.com/NMitchem/Sundial` (pushed 2026-07-19). The CI workflow (`.github/workflows/ci.yml`) **has** executed on GitHub runners and passed on its first run (3m20s, ubuntu-latest) — `RELEASE.md` §1–2 are effectively done. The repo is still **private**; making it public is the outstanding step.
- `npm publish` for `sundial-lp` has never been run anywhere — the package exists only as a local, inspected `.tgz` (see M2 results); publishing is a `RELEASE.md` step, not part of this milestone.

## M3 seeds (carried out of M2, for the next plan)

- **64×64 transport grid / in-shader cost generation:** the matrix-free transport operator materializes only the cost vector today (4 MiB at g=32); at g=64 (16.7M vars) that cost vector grows to 67 MiB, so cost generation needs to move in-shader before the hero can scale past 32×32.
- **df64-iterates decision:** revisit only when either (a) wgpu exposes an IEEE-strict/math-mode control for the Metal backend (no such control exists as of wgpu 30/naga 30 — tracked by watching wgpu releases), or (b) targeting a Vulkan/DX12-only context where fast-math isn't forced on by default. Even then, Finding 3's cross-workgroup f32 collapse (the Neumaier reduction combine) must also be fixed before a revisit could show a real precision-tier gain.
- ~~**Movement-based primal-weight update:** a `‖Δx‖/‖Δy‖`-style ω update for the explicit (Ruiz+PC-equilibrated) path.~~ **Done 2026-08-08, opt-in** (CPU + GPU; 20/32 → 30/32 on the GPU sweep, 5.2× fewer iterations on instances that already solved). Default deliberately **off** — see the M3 entries above for the accuracy trade that decision turns on.
- **GPU-presolve revisit condition:** once explicit-CSR-path benchmarking on Sundial specifically shows large/redundant instances where presolve-shaped headroom actually exists (the literature's win rate is concentrated in a minority of large/slow instances, not universal) — see `docs/notes/gpu-presolve-memo.md`.

Performance and tooling items carried forward, none blocking:

- **Alias original-space and iterate-space GPU buffers when scaling is identity.** The operator path currently uploads both; aliasing halves the n-sized buffer count at 1M+ variables.
- **Bind-group caching in `GpuOp` implementations.** Per-iteration bind-group creation is measurable CPU overhead at high iteration rates.
- **Comparison automation against published CPU/CUDA numbers.** `sundial report` links to the Mittelmann benchmark; a curated overlap table is manual today.
- **Chart: true partial redraw**, if rAF coalescing proves insufficient at hero scale.
