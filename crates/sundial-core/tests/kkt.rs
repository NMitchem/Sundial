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
