use sundial_core::problem::{CsrMatrix, LpProblem};
use sundial_core::{kkt, testgen};

#[test]
fn constructed_optimum_has_zero_residuals() {
    for seed in 0..20u64 {
        let (p, x, y, obj) = testgen::generate(seed, 40, 25);
        let r = kkt::residuals(&p, &x, &y);
        assert!(
            r.rel_primal < 1e-12,
            "seed {seed}: rel_primal={}",
            r.rel_primal
        );
        assert!(r.rel_dual < 1e-12, "seed {seed}: rel_dual={}", r.rel_dual);
        assert!(r.rel_gap < 1e-12, "seed {seed}: rel_gap={}", r.rel_gap);
        assert!((r.primal_obj - obj).abs() <= 1e-9 * (1.0 + obj.abs()));
    }
}

#[test]
fn perturbation_raises_residuals() {
    let (p, mut x, y, _) = testgen::generate(7, 40, 25);
    x[0] += 0.5;
    let r = kkt::residuals(&p, &x, &y);
    assert!(
        r.mu() > 1e-6,
        "perturbed point should not look optimal, mu={}",
        r.mu()
    );
}

#[test]
fn csr_transpose_roundtrip() {
    let (p, x, _, _) = testgen::generate(3, 30, 20);
    // (Aᵀ)ᵀ x == A x
    let att = p.at.transpose();
    let mut ax = vec![0.0; p.a.n_rows];
    let mut attx = vec![0.0; att.n_rows];
    p.a.mul(&x, &mut ax);
    att.mul(&x, &mut attx);
    for i in 0..ax.len() {
        assert!((ax[i] - attx[i]).abs() < 1e-12);
    }
}

#[test]
fn gap_stays_finite_with_noise_on_open_bounds() {
    // Standard-form-style LP: some rows/cols carry an open (infinite) bound on
    // one side while the true multiplier sits exactly at the other (e.g. an
    // "inactive" row with y*=0 and row_upper=+inf). A slightly-perturbed dual
    // that pushes such a multiplier across zero must NOT poison the dual
    // objective: D stays finite, rel_gap is a real (enforced) number, mu is
    // finite. (Old buggy formula: seed 42 with this exact perturbation drives
    // dual_obj to -inf — see task-9-report.md "Gap-enforcement fix" for the
    // RED evidence.)
    let (p, x, mut y, _) = testgen::generate(42, 40, 25);
    // perturb duals slightly so reduced costs / row multipliers pick up
    // wrong-sign noise against open bounds
    for yi in y.iter_mut() {
        *yi += 1e-9;
    }
    let r = kkt::residuals(&p, &x, &y);
    assert!(r.dual_obj.is_finite(), "dual_obj = {}", r.dual_obj);
    assert!(r.rel_gap.is_finite(), "rel_gap = {}", r.rel_gap);
    assert!(r.mu().is_finite());
}

#[test]
fn gap_stays_finite_with_negative_reduced_cost_on_open_upper_bound() {
    // Hand-built 1x1 LP exercising the column-side (bound_terms) guard in
    // kkt::residuals, mirroring gap_stays_finite_with_noise_on_open_bounds but
    // deterministically instead of via random perturbation. col_upper is open
    // (+inf) and y is chosen so the reduced cost g = c + Aᵀy = 1 + (-2) = -1 < 0,
    // which selects the g<0 branch that multiplies by col_upper. Without the
    // `p.col_upper[j].is_finite()` guard this yields -1 * +inf = -inf.
    let a = CsrMatrix {
        n_rows: 1,
        n_cols: 1,
        indptr: vec![0, 1],
        indices: vec![0],
        values: vec![1.0],
    };
    let p = LpProblem::new(
        "colside".into(),
        a,
        vec![1.0],               // c
        0.0,                     // obj_offset
        vec![f64::NEG_INFINITY], // row_lower
        vec![10.0],              // row_upper
        vec![0.0],               // col_lower
        vec![f64::INFINITY],     // col_upper (open upper bound)
    )
    .unwrap();
    let x = vec![1.0];
    let y = vec![-2.0];
    let r = kkt::residuals(&p, &x, &y);
    assert!(r.dual_obj.is_finite(), "dual_obj = {}", r.dual_obj);
    assert!(r.rel_gap.is_finite(), "rel_gap = {}", r.rel_gap);
    assert!(r.mu().is_finite());
}
