//! PDLP-style primal-weight balancing. Added because the 1M-variable
//! transport instance stalled under τ = σ, with the dual residual at ~1e-9
//! while the primal sat at ~1e-4 — badly unbalanced steps, not slow ones.
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

/// θ for the movement update's log-space smoothing (PDLP's default).
pub const THETA: f64 = 0.5;

/// Movement threshold below which a restart is treated as carrying no
/// information about step balance (guards 0/0 and denormal ratios).
const MOVEMENT_EPS: f64 = 1e-12;

/// PDLP's movement-based primal-weight update (Applegate et al., §5.2), applied
/// at restarts from the primal/dual movement between consecutive restart points:
///
/// ω ← exp(θ·log(Δy/Δx) + (1−θ)·log ω) = ω^(1−θ) · (Δy/Δx)^θ
///
/// The structural difference from [`update_primal_weight`] is the reason this
/// exists. The residual rule multiplies ω by the full √-damped ratio at every
/// restart, so it has NO fixed point unless the residuals balance exactly —
/// on the Ruiz+PC-equilibrated path that made it orbit between ω 0.02 and
/// 0.12. This is a contraction in log space toward log(Δy/Δx): the error halves
/// every restart, so a steady movement regime pins ω instead of orbiting it.
pub fn update_primal_weight_movement(omega: f64, dx: f64, dy: f64) -> f64 {
    if !(dx.is_finite() && dy.is_finite()) || dx <= MOVEMENT_EPS || dy <= MOVEMENT_EPS {
        return omega;
    }
    let target = dy / dx;
    if !target.is_finite() || target <= 0.0 {
        return omega;
    }
    (omega.powf(1.0 - THETA) * target.powf(THETA)).clamp(OMEGA_MIN, OMEGA_MAX)
}
