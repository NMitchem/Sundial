# M2 backlog (from M1, 2026-07-07)

Carried from the M1 plan and the whole-milestone review; none block M1.

- In-shader transport cost (g=64 / 16.7M vars needs it; cost vector is 67 MiB there) — adjudicated out of M1.
- Alias original-space and iterate-space GPU buffers when scaling is identity (op path currently uploads both; halves n-sized buffer count at 1M+).
- Bind-group caching in GpuOp implementations (per-iteration creation is measurable CPU overhead at high iteration rates).
- Comparison automation against published CPU/CUDA numbers (report links plato; a curated overlap table is manual today).
- Chart: true partial redraw if rAF coalescing proves insufficient at hero scale.
- Movement-based (‖Δx‖/‖Δy‖) primal-weight update if the explicit (Ruiz+PC-equilibrated) path ever needs ω — the residual-ratio version limit-cycles there (Task 6a adjudication).
- blend.mps RHS format support (set-name-less RHS lines — real-world netlib corner; currently an honest Error row).
- up_negative flag not reset on repeated same-column UP lines (contrived input; one-line hardening).
- e226 objective-constant sign-convention: consider a per-instance note column or alternate-convention value in the optima data (do NOT silently change values — see README/STATUS footnote).
- Scrub '|' from error text before markdown table rendering (report.rs).

## M3 code-hygiene (from M2 final review)

- `crates/sundial-core/src/transport.rs:211` — `.map(|&x| if x.is_finite() && x > 0.0 { x } else { 0.0 }.max(1e-9))`: without parens, `.max(1e-9)` binds only to the else-branch's `0.0`, not the whole if/else (harmless today since the else branch is the only path that needs the floor, but add parens + a comment so it doesn't silently break if the branches ever change).
- Checker preset parity expression duplicated: `transport.rs`'s `(Preset::Checker, true)` and `(Preset::Checker, false)` arms both recompute `(px * 4.0).floor() as i64 + (py * 4.0).floor() as i64) % 2 == 0` instead of computing the parity bool once and inverting it for the second arm.
- GPU-vs-CPU dual-projection asymmetry on certified exits (status-neutral, documented): on a certified Infeasible/Unbounded exit, the GPU engine (`gpu/engine.rs`) sign-projects the returned dual via `project_dual` before reporting it; the CPU reference path (`reference.rs`) returns the raw unscaled `y` on the same exits. Doesn't affect status honesty (both are already gated by the f64 Farkas check), but the returned dual differs by engine.
- Detection re-unscale redundancy: `reference.rs`'s restart-cadence block calls `unscale(scaling, &st.x/&st.y, ...)` twice per check when a streak is live — once for `r_cur`'s residuals, again a few lines later for the divergence-detection norms — recomputing the identical unscale on the same vectors.
