//! Farkas / recession-ray certificate verification (spec M2 §1). Engines
//! only FLAG candidates (iterate-norm growth streaks at restarts); a status
//! of Infeasible/Unbounded is set exclusively after one of these f64 checks
//! passes — the same certificate honesty as Optimal. See the plan's
//! "Certificate math" block for the derivations.
use crate::problem::LpView;

pub const EPS_RAY: f64 = 1e-6;
pub const EPS_GAIN: f64 = 1e-6;
/// Consecutive restarts of monotonic norm growth + stalled residual before
/// a (readback +) verification attempt.
pub const STREAK_K: u32 = 3;
/// Per-restart iterate-norm growth factor that flags a divergence candidate.
/// Infeasible/unbounded PDHG diverges LINEARLY, not geometrically: the iterate
/// difference converges to a fixed ray (Applegate et al. 2021), so ‖iterate‖
/// ≈ c·k and the per-restart ratio (k+1)/k decays toward 1. This is therefore
/// a monotonic-growth-with-noise-margin predicate — deliberately just above 1,
/// so the streak sustains through many restarts and verification retries as the
/// ray cleans up (a geometric threshold like 1.5 only holds for the first ~2
/// restarts and misses detection). The f64 verification remains the ONLY thing
/// that can set an Infeasible/Unbounded status; this constant just gates when a
/// verification is attempted, so it cannot produce a false status.
pub const GROWTH: f64 = 1.02;
pub const STALL: f64 = 0.9;

/// Verify a primal-infeasibility certificate from a dual ray candidate
/// (any scale — this fn sign-projects against open row bounds, then
/// normalizes). Returns the certified ray gain D₀(ŷ) when
/// ‖r_d,0(ŷ)‖ ≤ EPS_RAY and D₀(ŷ) ≥ EPS_GAIN.
pub fn verify_infeasible(orig: &LpView, y: &[f64]) -> Option<f64> {
    let (m, n) = (orig.op.n_rows(), orig.op.n_cols());
    assert_eq!(y.len(), m);
    // A valid ray never pushes on an open bound; f32 noise does. Projecting
    // first means noise can only shrink the candidate, never fake a gain.
    let mut ray: Vec<f64> = y.to_vec();
    for (i, ri) in ray.iter_mut().enumerate() {
        if !orig.row_upper[i].is_finite() && *ri > 0.0 {
            *ri = 0.0;
        }
        if !orig.row_lower[i].is_finite() && *ri < 0.0 {
            *ri = 0.0;
        }
    }
    let norm = ray.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        return None;
    }
    ray.iter_mut().for_each(|v| *v /= norm);
    // c≡0 view at x = 0: rel_dual IS ‖r_d,0(ŷ)‖ (c_norm = 0 ⇒ denominator 1)
    // and dual_obj IS D₀(ŷ) (obj_offset = 0).
    let zeros_c = vec![0.0; n];
    let view0 = LpView {
        op: orig.op,
        c: &zeros_c,
        obj_offset: 0.0,
        row_lower: orig.row_lower,
        row_upper: orig.row_upper,
        col_lower: orig.col_lower,
        col_upper: orig.col_upper,
    };
    let x0 = vec![0.0; n];
    let r = crate::kkt::residuals_view(&view0, &x0, &ray);
    (r.rel_dual <= EPS_RAY && r.dual_obj >= EPS_GAIN).then_some(r.dual_obj)
}

/// Verify an unboundedness certificate from a primal ray candidate (any
/// scale — normalized here; NOT projected: a candidate that violates the
/// column recession cone is judged invalid, not repaired). Returns |cᵀx̂|
/// when the recession conditions hold and cᵀx̂ ≤ −EPS_GAIN.
pub fn verify_unbounded(orig: &LpView, x: &[f64]) -> Option<f64> {
    let (m, n) = (orig.op.n_rows(), orig.op.n_cols());
    assert_eq!(x.len(), n);
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        return None;
    }
    let ray: Vec<f64> = x.iter().map(|v| v / norm).collect();
    // Column recession conditions (kkt's primal residual covers rows only).
    for (j, &rj) in ray.iter().enumerate() {
        if orig.col_lower[j].is_finite() && rj < -EPS_RAY {
            return None;
        }
        if orig.col_upper[j].is_finite() && rj > EPS_RAY {
            return None;
        }
    }
    // Recession view: finite bounds ↦ 0, open bounds stay ±∞. Its q_norm is
    // 0, so rel_primal IS the row-recession residual norm; primal_obj IS cᵀx̂.
    let map0 = |b: &f64| if b.is_finite() { 0.0 } else { *b };
    let rl: Vec<f64> = orig.row_lower.iter().map(map0).collect();
    let ru: Vec<f64> = orig.row_upper.iter().map(map0).collect();
    let cl: Vec<f64> = orig.col_lower.iter().map(map0).collect();
    let cu: Vec<f64> = orig.col_upper.iter().map(map0).collect();
    let view_rec = LpView {
        op: orig.op,
        c: orig.c,
        obj_offset: 0.0,
        row_lower: &rl,
        row_upper: &ru,
        col_lower: &cl,
        col_upper: &cu,
    };
    let y0 = vec![0.0; m];
    let r = crate::kkt::residuals_view(&view_rec, &ray, &y0);
    (r.rel_primal <= EPS_RAY && r.primal_obj <= -EPS_GAIN).then_some(-r.primal_obj)
}
