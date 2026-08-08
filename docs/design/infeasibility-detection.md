# Infeasibility and unboundedness detection

The design document (2026-07-10) for verified `Infeasible`/`Unbounded`
detection, the df64 precision experiment, and the npm packaging that followed
[`architecture.md`](architecture.md). Section 1 carries the Farkas certificate
math that `sundial-core/src/farkas.rs` implements.

## Goal

Make Sundial launch-ready without publishing anything: verified infeasibility detection, the df64 precision experiment (measured honestly, win or lose), hero-page interactivity (draw-your-own + presets), the Show HN writeup, a publish checklist — and the npm package as the very last tasks (user decision 2026-07-07). "Click submit" remains a human act after M2.

**Non-goals (deferred post-launch):** 64×64 grids / in-shader cost generation; identity-scaling buffer aliasing; bind-group caching; movement-based (‖Δx‖/‖Δy‖) primal weight for the explicit path; chart partial redraw; automated comparison-table generation; df64 *iterates* (only accumulators are in scope); actually publishing (GitHub push, Pages deploy, `npm publish`).

## Decisions log

| Decision | Choice | Why |
|---|---|---|
| M2 endpoint | Launch-ready, NOTHING published | User decision 2026-07-10: all artifacts finished locally (site built, writeup drafted, `npm pack` clean); user pushes the publish buttons |
| npm package position | Very END of the plan | User decision 2026-07-07 |
| df64 success bar | Timeboxed experiment, Tier-1 (accumulators) only | Measured numbers + memo are the deliverable, whatever the outcome; df64 iterates gated on Tier-1 results (M3 decision) |
| Infeasibility depth | Verified detection; no ray in public API | GPU flags candidates only; CPU-f64 Farkas verification before any `Infeasible`/`Unbounded` status — same certificate honesty as `Optimal`. Missed detection degrades to today's honest `IterationLimit` |
| Hero polish scope | Draw-your-own + 3 new presets; 64×64 OUT | User selection 2026-07-10; 16.7M-var headline deferred with its in-shader-cost prerequisite |
| npm crate | New `crates/sundial-web` (cdylib); wasm bindings move out of sundial-mps | sundial-mps was never the right home; parser crate returns to pure parser |
| npm name | `sundial-lp`, fallback `@sundial/solver` | Availability checked at execution time via `npm view` — never assumed |
| Backlog triage IN | blend.mps RHS fix, `\|` scrub in report, up_negative reset, e226 note annotation | Small, launch-visible (table quality) |
| Comparison table | Manually curated for the writeup | Automation is post-launch backlog; overlap with plato-published instances is small |
| GPU-presolve paper (arXiv 2604.23951) | Read-and-decide memo, timeboxed | Spec'd as "evaluated"; outcome is an M3 recommendation, not implementation |

## Component designs

### 1. Infeasibility / unboundedness detection (`sundial-core`)

**Principle:** the GPU (and CPU reference) only *flag candidates*; the status is set exclusively after a CPU-f64 certificate check. False claims are structurally impossible; false negatives fall back to `IterationLimit` (today's behavior).

**Detection trigger (both engines, at restart cadence):**
- Track ‖y‖∞ (GPU: one more reduction into the free `results` slot 11; CPU: direct) and ‖x‖∞ (already tracked as the NaN watchdog).
- Candidate **primal-infeasibility** when, across K consecutive restarts (K=3), ‖y‖∞ grows geometrically (factor ≥ γ per restart, γ=1.5) while the primal residual has not improved by ≥10%. Candidate **unboundedness**: symmetric on ‖x‖∞ with the dual residual.

**Certificate extraction + f64 verification (host):**
- Read back the candidate iterate; normalize the ray (ŷ = y/‖y‖₂ or x̂ = x/‖x‖₂) in f64.
- **Farkas (infeasible):** using the existing kkt machinery evaluated with c ≡ 0 in ORIGINAL space: every component of Aᵀŷ must be absorbable by a finite variable bound (the c≡0 dual residual ≤ ε_ray), and the c≡0 dual objective (projected-multiplier convention) must be ≥ ε_gain. Then `SolveStatus::Infeasible`.
- **Improving ray (unbounded):** x̂ in the recession cone (rows with a finite bound satisfy the corresponding sign condition on (Ax̂); variable bounds likewise) with cᵀx̂ ≤ −ε_gain. Then `SolveStatus::Unbounded`.
- ε_ray = 1e-6 · scale, ε_gain = 1e-6 (exact values fixed in the plan; conservative — tightening them only trades detection speed for certainty).
- The Farkas ray is NOT exposed in the public API/Solution (decision above); it exists transiently for verification.

**Surface changes:** `SolveStatus::{Infeasible, Unbounded}` variants; CLI human/JSON output, bench CSV, `sundial report`, and wasm status strings all render them; bench/report treat them as honest terminal rows (not errors).

**Tests:** constructed oracles (infeasible: rows forcing x ≥ 1 and x ≤ 0; unbounded: min −x with x free above), detection within an iteration budget on both engines; a netlib `infeas` subset via an extended fetch script (execution-time download, same honesty rules as M1's fetch); and the critical gate — **zero false positives**: every existing feasible suite (Netlib 3/3+5/5, transport, testgen) must still terminate `Optimal`/`IterationLimit` exactly as before.

### 2. df64 experiment (`sundial-core`, timeboxed)

**Scope (Tier 1 only):** double-double (TwoSum/TwoProd, Dekker/Knuth) accumulation inside: CSR SpMV row loops, `ot_apply` row/col sums, and the `reduce_dot`/`reduce_sum` kernels (upgrading Neumaier compensation to full df64 partial sums). Iterates, bounds, and all other buffers stay f32. WGSL stays portable-subset (df64 is plain f32 ops).

**Switch:** `SolveOptions::df64: bool` (default false) → kernel-variant selection at pipeline build; CLI `--df64`. Default paths are byte-identical to M1 when off.

**Protocol (fixed before running; results reported whatever they say):**
1. The 12 M1 f32-wall instances at tol 1e-4, `--df64`: how many now reach verified Optimal?
2. The 19 M1 passers at tol 1e-6, `--df64` and without: residual floor per instance; how many reach verified 1e-6?
3. Hero (g=32) with `--df64`: achieved floor + per-iteration cost delta.

**Deliverable:** `docs/notes/df64-experiment.md` (protocol, raw numbers, honest interpretation) + a writeup section. Decision gate recorded there: df64 iterates go to M3 consideration only if Tier 1 gains ≥3 instances a tier.

### 3. Hero polish (`web/` + small core/wasm additions)

- **Presets:** add `spiral`, `checker`, `corners` (exact density formulas fixed at plan time; same floor-at-1e-9 + normalize pipeline; CPU tests for normalization/positivity as in M1).
- **Draw-your-own:** pointer-event brush painting directly on the source/target canvases (brush radius + intensity, additive with clamp, per-canvas clear); a preset ↔ draw mode toggle; Solve uses the painted masses.
- **wasm:** `solveTransportCustom(grid, src: Float32Array, tgt: Float32Array, tol, onProgress, onSnapshot)`; core: `transport::problem_from_masses(src, tgt, g)` — validates lengths, floors at 1e-9, normalizes both sides to sum exactly 1 (feasible by construction, like presets). Painting an all-zero canvas is valid (floor makes it uniform).

### 4. Backlog hygiene (launch-visible fixes)

- **blend.mps RHS format:** support set-name-less RHS lines (even-token RHS rows parsed as `row val` pairs; line-numbered error retained for genuinely malformed rows). Re-run the sweep row: table target becomes 32 parsed / ≥19 Optimal.
- **`|` scrub** in report.rs error text before markdown rendering.
- **up_negative reset:** `up_negative[j] = val < 0.0` unconditionally in the UP arm (repeated-UP hardening) + a test.
- **e226 annotation:** optional third CSV column `note` in `netlib_optima.csv`; `sundial report` renders a footnote line for any instance carrying a note. e226 gets the sign-convention note; values untouched.

### 5. npm package (`crates/sundial-web`, the FINAL tasks)

- New cdylib crate `sundial-web`; move + re-export the wasm API from sundial-mps (which drops its `wasm.rs`, cdylib crate-type, and wasm-only deps); demo pages import the new pkg path.
- Package name `sundial-lp` — checked at execution with `npm view sundial-lp` (and fallback `@sundial/solver` if taken); name recorded in the plan run, never assumed.
- API surface: `solveMps(text)`, `solveMpsBytes(bytes)`, `solveTransport(...)`, `solveTransportCustom(...)`, `transportPreview(...)`, `webgpuAvailable(): boolean` (capability probe, no GPU init); typed result objects via a hand-maintained `.d.ts` refinement layered over wasm-pack output; package README with a copy-paste browser example and the honest-limits paragraph.
- Verification: `wasm-pack build --target web`, `npm pack` producing a clean tarball; the demo site builds against the packed output (proves the package is self-sufficient). **No `npm publish`.**

### 6. Writeup + launch readiness (local artifacts only)

- **`docs/writeup.md`** — the Show HN / blog draft: CUDA lock-in framing → WebGPU portability; certificate honesty (GPU never grades its own homework; f64 verification; projected-multiplier gap); the primal-weight war story (500k-iteration stall → 16k, with the residual-curve figure); hero numbers (native ~9.4 s; browser 0.547 ms/iter, user-confirmed); df64 experiment results; the curated comparison table (our verified numbers beside plato-published CPU/CUDA times for overlapping instances, with explicit caveats about tolerance tiers and hardware class); honest-limits section (f32/1e-4 default tier, no presolve, where CPU simplex wins, 12/32 wall instances).
- **`RELEASE.md`** — the user's publish checklist: create the public GitHub repo (note: local dir `or-fable`, Cargo.toml says `sundial` — pick the public name then), push, first CI run expectations (CI has never executed on GitHub runners), enable Pages + deploy `web/dist`, `npm publish` steps, Show HN title options (from the proposal doc).
- **Site:** production `vite build` verified from a clean checkout; the no-WebGPU explainer already ships.
- **GPU-presolve memo:** read arXiv 2604.23951, one-page `docs/notes/gpu-presolve-memo.md` with an adopt/defer/reject recommendation for M3. Timeboxed; no implementation.

## Error handling

| Case | Behavior |
|---|---|
| Infeasible input | `Infeasible` only after f64 Farkas verification; otherwise falls through to `IterationLimit` (never a false claim) |
| Unbounded input | Same pattern via primal ray |
| Painted masses degenerate (all zero) | Floor + normalize ⇒ uniform distribution; never an error |
| npm name taken | Fallback name; if both taken, STOP and ask the user |
| df64 shows no gains | Honest memo; experiment is still complete (the bar is measurement, not victory) |
| netlib infeas set unreachable | Same rule as M1: STOP and report; never substitute unverified mirrors |

## Testing & gates

- All M1 suites stay green at every task boundary (CPU, GPU incl. 1M gate, Netlib gates, fmt, clippy native+wasm32, wasm32 build, wasm-pack, tsc, vite build).
- New gates: infeasibility oracles (both engines) + zero-false-positive sweep; df64 on/off byte-identical when off; preset/custom-mass normalization tests; parser blend fixture test (real netlib file); report note-column and `|`-scrub unit tests; `npm pack` tarball contents check.
- Browser human gate at close: draw-your-own solve + one new preset, user-confirmed (dev server from the user's own terminal — in-session servers get SIGTERM'd).

## Exit criteria (M2 done =)

1. Verified infeasibility/unboundedness detection on both engines; zero false positives across all feasible suites.
2. df64 memo committed with the full measured protocol (any outcome).
3. Hero: draw-your-own + ≥5 total presets, user-confirmed in browser.
4. Netlib table: all 32 instances parse (blend fixed); honest statuses; e226 footnoted in the rendered report.
5. `sundial-web` npm package: `npm pack` clean, README + types, name availability recorded. NOT published.
6. `docs/writeup.md` complete incl. comparison table + honest limits; `RELEASE.md` publish checklist complete.
7. GPU-presolve memo committed.
8. Full verification suite green; STATUS.md M2 record written.
