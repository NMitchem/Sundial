use sundial_core::farkas::{verify_infeasible, verify_unbounded};
use sundial_core::problem::{CsrMatrix, LpProblem, SolveOptions, SolveStatus};
use sundial_core::{reference, testgen};

/// x must satisfy x ≥ 1 (row 0) and x ≤ 0 (row 1): infeasible.
fn infeasible_lp() -> LpProblem {
    let a = CsrMatrix {
        n_rows: 2,
        n_cols: 1,
        indptr: vec![0, 1, 2],
        indices: vec![0, 0],
        values: vec![1.0, 1.0],
    };
    LpProblem::new(
        "infeas-2row".into(),
        a,
        vec![0.0],
        0.0,
        vec![1.0, f64::NEG_INFINITY],
        vec![f64::INFINITY, 0.0],
        vec![f64::NEG_INFINITY],
        vec![f64::INFINITY],
    )
    .unwrap()
}

/// min −x subject to x ≥ 0 (row) with x ∈ [0, ∞): unbounded below.
fn unbounded_lp() -> LpProblem {
    let a = CsrMatrix {
        n_rows: 1,
        n_cols: 1,
        indptr: vec![0, 1],
        indices: vec![0],
        values: vec![1.0],
    };
    LpProblem::new(
        "unbounded-1var".into(),
        a,
        vec![-1.0],
        0.0,
        vec![0.0],
        vec![f64::INFINITY],
        vec![0.0],
        vec![f64::INFINITY],
    )
    .unwrap()
}

#[test]
fn handmade_farkas_ray_verifies() {
    let p = infeasible_lp();
    // ŷ = (−1, +1)/√2: row0 active-lower (y<0), row1 active-upper (y>0);
    // Aᵀŷ = 0 (absorbed trivially), D₀ = −(1·(−1/√2) + 0·(1/√2)) = 1/√2 > 0.
    let gain = verify_infeasible(&p.view(), &[-1.0, 1.0]).expect("valid certificate");
    assert!((gain - 1.0 / 2f64.sqrt()).abs() < 1e-12, "gain {gain}");
}

#[test]
fn wrong_sign_ray_rejected() {
    let p = infeasible_lp();
    // (+1, −1) pushes on the OPEN sides; project_dual zeroes it entirely.
    assert!(verify_infeasible(&p.view(), &[1.0, -1.0]).is_none());
    assert!(verify_infeasible(&p.view(), &[0.0, 0.0]).is_none());
}

#[test]
fn feasible_problem_rejects_rays() {
    let (p, _, y, _) = testgen::generate(3, 20, 12);
    // a feasible problem's optimal dual is NOT a Farkas ray
    assert!(verify_infeasible(&p.view(), &y).is_none());
}

#[test]
fn handmade_improving_ray_verifies() {
    let p = unbounded_lp();
    let gain = verify_unbounded(&p.view(), &[1.0]).expect("valid improving ray");
    assert!((gain - 1.0).abs() < 1e-12, "gain {gain}"); // |cᵀx̂| = 1
}

#[test]
fn recession_violating_ray_rejected() {
    let p = unbounded_lp();
    assert!(verify_unbounded(&p.view(), &[-1.0]).is_none()); // col lower finite: x̂ < 0 invalid
}

#[test]
fn cpu_detects_infeasible() {
    let p = infeasible_lp();
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = reference::solve(&p, &opts, &mut |_| {});
    assert_eq!(
        sol.status,
        SolveStatus::Infeasible,
        "iters {}",
        sol.stats.iterations
    );
}

#[test]
fn cpu_detects_unbounded() {
    let p = unbounded_lp();
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = reference::solve(&p, &opts, &mut |_| {});
    assert_eq!(
        sol.status,
        SolveStatus::Unbounded,
        "iters {}",
        sol.stats.iterations
    );
}

#[test]
fn no_false_positives_on_feasible_set() {
    // the merge gate in miniature: constructed-feasible instances never
    // come back Infeasible/Unbounded
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 300_000,
        ..Default::default()
    };
    for seed in 0..4u64 {
        let (p, _, _, _) = testgen::generate(seed, 30, 20);
        let sol = reference::solve(&p, &opts, &mut |_| {});
        assert!(
            matches!(
                sol.status,
                SolveStatus::Optimal | SolveStatus::IterationLimit
            ),
            "seed {seed}: {:?}",
            sol.status
        );
    }
}
