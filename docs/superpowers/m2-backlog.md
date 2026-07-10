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
