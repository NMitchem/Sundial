# Sundial — Project Status

Updated: 2026-07-07 (M0 completion)

## Milestones

- [x] **M0** — GPU solver core (restarted PDHG in WGSL), MPS parser, CPU f64 reference, CLI, Netlib gate, minimal web demo. **Merged to main 2026-07-07.**
- [ ] **M1** (spec: weeks 2–4) — matrix-free `LinOp`; **optimal-transport hero at ≥1M variables, interactive in-browser**; drop-a-file benchmark page; CLI sweeps producing the comparison table vs published CPU/CUDA numbers. Prerequisite from final review: batch the per-check GPU readbacks.
- [ ] **M2** (spec: weeks 5–8) — npm package; hero polish (presets + draw-your-own); double-double precision experiment; infeasibility detection; launch writeup ("Show HN").

**Launch bar** (from spec): hero ≥1M vars → 1e-4 interactive on a MacBook; ≥25 Netlib + ≥5 large instances in the table; `npm install` works; writeup done.

## M0 results (verified on Apple M4 Pro / Metal, 2026-07-07)

- **GPU Netlib gate: 5/5** fixtures (afiro, sc50a, sc50b, adlittle, share2b) reach `Optimal` at relative KKT ≤ 1e-4, CPU-f64-verified, with relative objective error ≤ 1e-3 vs published optima (worst case 6.7e-4).
- **CPU reference gate (CI): 3/3** (afiro, sc50a, sc50b) at the same tolerances.
- Example run: afiro on GPU — objective −464.7530946 (published −464.75314286), 4,352 iterations, 5 restarts, ~760 ms wall including per-check readback overhead (readback dominates at this tiny scale; the GPU pays off at M1 sizes).
- Suites: 18 CPU tests + 8 GPU tests (`#[ignore]`d in CI, run locally with `--include-ignored`); fmt / clippy (native + wasm32) / wasm32 build gates all green.

## Design decisions of record (full detail in spec + plan)

- **Certificate honesty:** `Optimal` is only ever set after a CPU f64 KKT recheck at the exact returned point; the dual is sign-projected first and the returned solution *is* the verified point.
- **Projected-multiplier dual objective on both row and column sides** — the duality gap stays finite and is genuinely enforced (an f64 ±inf would have turned it into NaN and silently dropped it from the certificate on every standard-form LP). Adjudicated mid-M0; plan commits 91608e5 / fcec2c0.
- **GPU infinity convention:** ±1e30 sentinels in every GPU buffer; shaders never perform inf arithmetic (browser fast-math portability).

## Key documents

- `docs/superpowers/specs/2026-07-07-sundial-design.md` — approved spec: architecture, milestones, launch bar, platform constraints
- `docs/superpowers/plans/2026-07-07-sundial-m0.md` — M0 plan (normative, includes all mid-flight adjudications)
- `docs/superpowers/plans/2026-07-07-sundial-m1.md` — M1 plan (12 tasks; written 2026-07-07, execution pending)
- `docs/superpowers/m1-backlog.md` — minor items carried from M0's final review
- `or-project-proposals.md` — the original OR project survey (Sundial plus 4 other adversarially-vetted proposals)

## Known gaps / notes

- No GitHub remote yet — the CI workflow (`.github/workflows/ci.yml`) has never executed on GitHub runners.
- Interactive browser verification of the web demo: pending user confirmation (dev server runs via `cd web && npm run dev`).
- Task-level execution records (briefs, implementer reports, review verdicts) were session scratch (`.superpowers/sdd/`, gitignored); their outcomes are summarized here and in the plan's adjudication commits.
