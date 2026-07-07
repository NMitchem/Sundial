# M1 backlog (from M0 final review, 2026-07-07)

Carried from the whole-branch review; none block M0.

- Parser: fixtures/tests for error paths (UP-negative bound policy, missing ENDATA, no N row); decide duplicate-ROWS-name handling (currently latent phantom row on malformed input).
- Core: direct tests for ProblemError::Dimension/BoundOrder; promote reference.rs final honesty debug_assert to hard assert (symmetry with engine); refresh project_dual doc comment (its current role is dual-residual cleanup; the -inf dual-objective case is already prevented by kkt.rs projection).
- Engine: batch the ~13 per-check readbacks (one packed staging readback) before the 1M-variable hero demo.
- CLI: quote/escape the bench CSV name field; print full anyhow chains ({e:#}) in bench error rows.
- Web: incremental chart drawing (currently full redraw per progress event); consider console.error for swallowed progress-callback exceptions.
- Buffers: size_of::<f32>() instead of magic 4 in readback (cosmetic).
