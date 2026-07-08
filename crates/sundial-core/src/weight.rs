//! PDLP-style primal-weight balancing (spec enhancement ladder step 3;
//! scoped into M1 by adjudication after the 1M transport gate stalled with
//! dual ~1e-9 / primal ~1e-4 under τ = σ).
//!
//! ω sets the τ/σ split with the step-size product invariant:
//! τ = 0.9/(‖A‖ω), σ = 0.9ω/‖A‖ ⇒ τσ‖A‖² = 0.81 for every ω.
//! ω > 1 pushes the DUAL side (feasibility); ω < 1 pushes the primal side.

pub const OMEGA_MIN: f64 = 1e-4;
pub const OMEGA_MAX: f64 = 1e4;

/// PDLP's ω₀ = ‖c‖/‖q‖ in ITERATE space: duals live at cost scale, primals
/// at rhs scale, so this ratio matches the step split to the problem's
/// natural scales. Degenerate norms (empty objective / fully-open rows)
/// fall back to the unweighted ω = 1.
/// NOTE argument order matches `kkt::denominators_view`: (q_norm, c_norm).
pub fn initial_primal_weight(q_norm: f64, c_norm: f64) -> f64 {
    if q_norm > 1e-12 && c_norm > 1e-12 {
        (c_norm / q_norm).clamp(OMEGA_MIN, OMEGA_MAX)
    } else {
        1.0
    }
}

/// Residual-balance update applied at every restart: a lagging primal
/// residual grows ω (push feasibility harder), a lagging dual shrinks it.
/// √ = θ = 0.5 log-space damping; degenerate ratios leave ω unchanged so
/// a converged side can never poison the weight.
pub fn update_primal_weight(omega: f64, rel_p: f64, rel_d: f64) -> f64 {
    let ratio = rel_p / rel_d;
    if !ratio.is_finite() || ratio <= 0.0 {
        return omega;
    }
    (omega * ratio.sqrt()).clamp(OMEGA_MIN, OMEGA_MAX)
}
