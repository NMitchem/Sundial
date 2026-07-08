use sundial_core::linop::{op_norm2, power_iteration_norm_op, CsrOp, LinOp};
use sundial_core::problem::{OpProblem, SolveOptions, SolveStatus};
use sundial_core::{kkt, reference, testgen};

/// Owned CSR pair so OpProblem can own its operator in tests.
struct OwnedCsr {
    a: sundial_core::problem::CsrMatrix,
    at: sundial_core::problem::CsrMatrix,
}
impl LinOp for OwnedCsr {
    fn n_rows(&self) -> usize {
        self.a.n_rows
    }
    fn n_cols(&self) -> usize {
        self.a.n_cols
    }
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        self.a.mul(x, out)
    }
    fn apply_t(&self, y: &[f64], out: &mut [f64]) {
        self.at.mul(y, out)
    }
}

#[test]
fn csrop_matches_matrix_mul() {
    let (p, _, _, _) = testgen::generate(3, 30, 20);
    let op = CsrOp { a: &p.a, at: &p.at };
    let mut rng = fastrand::Rng::with_seed(1);
    let x: Vec<f64> = (0..p.n_vars()).map(|_| rng.f64() - 0.5).collect();
    let y: Vec<f64> = (0..p.n_cons()).map(|_| rng.f64() - 0.5).collect();
    let (mut ax1, mut ax2) = (vec![0.0; p.n_cons()], vec![0.0; p.n_cons()]);
    op.apply(&x, &mut ax1);
    p.a.mul(&x, &mut ax2);
    assert_eq!(ax1, ax2);
    let (mut aty1, mut aty2) = (vec![0.0; p.n_vars()], vec![0.0; p.n_vars()]);
    op.apply_t(&y, &mut aty1);
    p.at.mul(&y, &mut aty2);
    assert_eq!(aty1, aty2);
}

#[test]
fn residuals_view_matches_residuals() {
    let (p, x, y, _) = testgen::generate(5, 40, 25);
    let direct = kkt::residuals(&p, &x, &y);
    let viewed = kkt::residuals_view(&p.view(), &x, &y);
    assert_eq!(direct.rel_primal, viewed.rel_primal);
    assert_eq!(direct.rel_dual, viewed.rel_dual);
    assert_eq!(direct.rel_gap, viewed.rel_gap);
    assert_eq!(direct.primal_obj, viewed.primal_obj);
}

#[test]
fn power_iteration_op_matches_matrix_version() {
    let (p, _, _, _) = testgen::generate(9, 35, 22);
    let via_op = power_iteration_norm_op(&CsrOp { a: &p.a, at: &p.at }, 100, 0);
    let via_mat = reference::power_iteration_norm(&p.a, &p.at, 100, 0);
    assert_eq!(via_op, via_mat); // same algorithm, same seed → bitwise equal
}

#[test]
fn norm2_exact_short_circuits() {
    struct Fixed;
    impl LinOp for Fixed {
        fn n_rows(&self) -> usize {
            1
        }
        fn n_cols(&self) -> usize {
            1
        }
        fn apply(&self, _: &[f64], out: &mut [f64]) {
            out[0] = 0.0
        }
        fn apply_t(&self, _: &[f64], out: &mut [f64]) {
            out[0] = 0.0
        }
        fn norm2_exact(&self) -> Option<f64> {
            Some(42.0)
        }
    }
    assert_eq!(op_norm2(&Fixed, 0), 42.0);
}

#[test]
fn solve_op_reaches_optimal_on_constructed_lp() {
    // testgen values are U(-2,2): already well-scaled, so the UNSCALED op
    // path (what matrix-free problems use) must converge on them.
    let (p, _, _, obj) = testgen::generate(2, 30, 20);
    let op = OpProblem::new(
        p.name.clone(),
        OwnedCsr {
            a: p.a.clone(),
            at: p.at.clone(),
        },
        p.c.clone(),
        p.obj_offset,
        p.row_lower.clone(),
        p.row_upper.clone(),
        p.col_lower.clone(),
        p.col_upper.clone(),
    )
    .unwrap();
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = reference::solve_op(&op, &opts, &mut |_| {});
    assert_eq!(sol.status, SolveStatus::Optimal);
    assert!(
        sol.stats.verified.mu() <= 1e-4,
        "mu={}",
        sol.stats.verified.mu()
    );
    let rel = (sol.primal_obj - obj).abs() / (1.0 + obj.abs());
    assert!(rel <= 1e-3, "obj {} vs {obj}", sol.primal_obj);
}

#[test]
fn op_problem_validates_dimensions() {
    let bad = OpProblem::new(
        "bad".into(),
        OwnedCsr {
            a: sundial_core::problem::CsrMatrix {
                n_rows: 2,
                n_cols: 3,
                indptr: vec![0, 0, 0],
                indices: vec![],
                values: vec![],
            },
            at: sundial_core::problem::CsrMatrix {
                n_rows: 3,
                n_cols: 2,
                indptr: vec![0, 0, 0, 0],
                indices: vec![],
                values: vec![],
            },
        },
        vec![0.0; 99],
        0.0,
        vec![0.0; 2],
        vec![0.0; 2],
        vec![0.0; 3],
        vec![0.0; 3],
    );
    assert!(bad.is_err());
}
