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
