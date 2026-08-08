use sundial_core::weight::{
    initial_primal_weight, update_primal_weight, update_primal_weight_movement, OMEGA_MAX,
    OMEGA_MIN,
};

#[test]
fn initial_weight_is_cost_over_rhs_scale() {
    assert_eq!(initial_primal_weight(0.1, 10.0), 100.0); // (q_norm, c_norm)
    assert_eq!(initial_primal_weight(10.0, 0.1), 0.01);
    assert_eq!(initial_primal_weight(1.0, 1.0), 1.0);
}

#[test]
fn initial_weight_degenerate_norms_fall_back_to_one() {
    assert_eq!(initial_primal_weight(0.0, 5.0), 1.0);
    assert_eq!(initial_primal_weight(5.0, 0.0), 1.0);
    assert_eq!(initial_primal_weight(0.0, 0.0), 1.0);
}

#[test]
fn initial_weight_clamps() {
    assert_eq!(
        initial_primal_weight(1e-10_f64.sqrt(), 1e10_f64.sqrt()),
        OMEGA_MAX
    );
    assert_eq!(
        initial_primal_weight(1e10_f64.sqrt(), 1e-10_f64.sqrt()),
        OMEGA_MIN
    );
}

#[test]
fn update_grows_omega_when_primal_lags() {
    // primal residual 1e-4, dual 1e-9 (the observed 1M-gate stall):
    // ratio 1e5, sqrt-damped to ~316x growth
    let w = update_primal_weight(1.0, 1e-4, 1e-9);
    assert!((w - (1e5_f64).sqrt()).abs() < 1e-6, "w={w}");
}

#[test]
fn update_shrinks_omega_when_dual_lags() {
    let w = update_primal_weight(1.0, 1e-9, 1e-4);
    assert!((w - (1e-5_f64).sqrt()).abs() < 1e-12, "w={w}");
}

#[test]
fn update_is_identity_on_balance_and_degenerate_inputs() {
    assert_eq!(update_primal_weight(3.0, 1e-6, 1e-6), 3.0); // balanced
    assert_eq!(update_primal_weight(3.0, 1e-6, 0.0), 3.0); // ratio inf
    assert_eq!(update_primal_weight(3.0, 0.0, 0.0), 3.0); // ratio NaN
    assert_eq!(update_primal_weight(3.0, f64::NAN, 1e-6), 3.0);
}

#[test]
fn update_clamps() {
    assert_eq!(update_primal_weight(OMEGA_MAX, 1e-2, 1e-9), OMEGA_MAX);
    assert_eq!(update_primal_weight(OMEGA_MIN, 1e-9, 1e-2), OMEGA_MIN);
}

// ---- movement-based ω (experimental) ----

#[test]
fn movement_update_contracts_toward_the_movement_ratio() {
    // THE crux of this rule. The residual-ratio update multiplies ω by the full
    // damped ratio at every restart, so it has no fixed point unless the
    // residuals happen to balance — the suspected mechanism behind the limit
    // cycle observed on the equilibrated path (ω orbited 0.02↔0.12).
    // The movement rule is a contraction in log space: under a steady movement
    // ratio ω must CONVERGE to it, not orbit it.
    let (dx, dy) = (1.0, 8.0); // target ω = Δy/Δx = 8
    let mut w = 1e-3;
    for _ in 0..60 {
        w = update_primal_weight_movement(w, dx, dy);
    }
    assert!((w - 8.0).abs() < 1e-9, "w={w}");
}

#[test]
fn movement_update_is_the_geometric_mean_of_omega_and_the_ratio() {
    // θ = 0.5 ⇒ one step lands exactly on √(ω · Δy/Δx)
    let w = update_primal_weight_movement(2.0, 1.0, 8.0);
    assert!((w - 16.0f64.sqrt()).abs() < 1e-12, "w={w}");
}

#[test]
fn movement_update_is_identity_on_degenerate_movement() {
    // a restart that moved nothing carries no information about step balance
    assert_eq!(update_primal_weight_movement(3.0, 0.0, 1.0), 3.0);
    assert_eq!(update_primal_weight_movement(3.0, 1.0, 0.0), 3.0);
    assert_eq!(update_primal_weight_movement(3.0, f64::NAN, 1.0), 3.0);
    assert_eq!(update_primal_weight_movement(3.0, 1.0, f64::INFINITY), 3.0);
}

#[test]
fn movement_update_clamps() {
    assert_eq!(
        update_primal_weight_movement(OMEGA_MAX, 1e-8, 1.0),
        OMEGA_MAX
    );
    assert_eq!(
        update_primal_weight_movement(OMEGA_MIN, 1.0, 1e-8),
        OMEGA_MIN
    );
}
