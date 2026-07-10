use sundial_core::linop::{power_iteration_norm_op, CsrOp, LinOp};
use sundial_core::problem::{SolveOptions, SolveStatus};
use sundial_core::reference;
use sundial_core::transport::{self, Preset, TransportOp};

#[test]
fn op_matches_explicit_csr() {
    let (ns, nt) = (36, 36); // g=6
    let op = TransportOp { ns, nt };
    let a = transport::explicit_csr(ns, nt);
    let at = a.transpose();
    let mut rng = fastrand::Rng::with_seed(11);
    let x: Vec<f64> = (0..ns * nt).map(|_| rng.f64() - 0.5).collect();
    let y: Vec<f64> = (0..ns + nt).map(|_| rng.f64() - 0.5).collect();
    let (mut ax_op, mut ax_csr) = (vec![0.0; ns + nt], vec![0.0; ns + nt]);
    op.apply(&x, &mut ax_op);
    a.mul(&x, &mut ax_csr);
    for (u, v) in ax_op.iter().zip(&ax_csr) {
        assert!((u - v).abs() <= 1e-12, "{u} vs {v}");
    }
    let (mut aty_op, mut aty_csr) = (vec![0.0; ns * nt], vec![0.0; ns * nt]);
    op.apply_t(&y, &mut aty_op);
    at.mul(&y, &mut aty_csr);
    for (u, v) in aty_op.iter().zip(&aty_csr) {
        assert!((u - v).abs() <= 1e-12, "{u} vs {v}");
    }
}

#[test]
fn exact_norm_matches_power_iteration() {
    let (ns, nt) = (25, 25); // g=5
    let exact = TransportOp { ns, nt }.norm2_exact().unwrap();
    assert_eq!(exact, ((ns + nt) as f64).sqrt());
    let a = transport::explicit_csr(ns, nt);
    let at = a.transpose();
    let est = power_iteration_norm_op(&CsrOp { a: &a, at: &at }, 300, 0);
    assert!(
        (exact - est).abs() / exact <= 1e-6,
        "exact {exact} vs est {est}"
    );
}

#[test]
fn all_presets_normalized_and_positive() {
    use sundial_core::transport::Preset::*;
    for preset in [Blobs, Ring, Spiral, Checker, Corners] {
        let (src, tgt) = transport::masses(preset, 16);
        for (side, v) in [("src", &src), ("tgt", &tgt)] {
            assert_eq!(v.len(), 256, "{preset:?} {side}");
            assert!(
                (v.iter().sum::<f64>() - 1.0).abs() < 1e-12,
                "{preset:?} {side}"
            );
            assert!(v.iter().all(|&x| x > 0.0), "{preset:?} {side}");
        }
    }
}

#[test]
fn new_preset_names_parse() {
    for (s, want) in [
        ("spiral", transport::Preset::Spiral),
        ("checker", transport::Preset::Checker),
        ("corners", transport::Preset::Corners),
    ] {
        assert_eq!(s.parse::<transport::Preset>().unwrap(), want);
    }
    assert!("nope".parse::<transport::Preset>().is_err());
}

#[test]
fn problem_dimensions_and_bounds() {
    let p = transport::problem(Preset::Blobs, 4);
    assert_eq!(p.n_vars(), 256); // (4²)²
    assert_eq!(p.n_cons(), 32); // 2·4²
    assert_eq!(p.row_lower, p.row_upper); // all equality rows
    assert!(p.col_lower.iter().all(|&l| l == 0.0));
    assert!(p.col_upper.iter().all(|&u| u == f64::INFINITY));
    assert!(p.c.iter().all(|&c| (0.0..=2.0).contains(&c)));
}

#[test]
fn solve_op_matches_explicit_solve_g4() {
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let op_p = transport::problem(Preset::Blobs, 4);
    let sol_op = reference::solve_op(&op_p, &opts, &mut |_| {});
    assert_eq!(sol_op.status, SolveStatus::Optimal);
    let ex_p = transport::explicit_problem(Preset::Blobs, 4);
    let sol_ex = reference::solve(&ex_p, &opts, &mut |_| {});
    assert_eq!(sol_ex.status, SolveStatus::Optimal);
    let rel = (sol_op.primal_obj - sol_ex.primal_obj).abs() / (1.0 + sol_ex.primal_obj.abs());
    assert!(
        rel <= 1e-3,
        "op {} vs explicit {}",
        sol_op.primal_obj,
        sol_ex.primal_obj
    );
}

#[test]
fn identical_marginals_cost_zero() {
    // src == tgt ⇒ X = diag(src) is optimal with objective exactly 0
    // (diagonal cost is 0). Exact oracle with no external solver.
    let (src, _) = transport::masses(Preset::Blobs, 4);
    let ns = src.len();
    let mut row = src.clone();
    row.extend_from_slice(&src);
    let p = sundial_core::problem::OpProblem::new(
        "self-transport".into(),
        TransportOp { ns, nt: ns },
        transport::cost_vector(4),
        0.0,
        row.clone(),
        row,
        vec![0.0; ns * ns],
        vec![f64::INFINITY; ns * ns],
    )
    .unwrap();
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = reference::solve_op(&p, &opts, &mut |_| {});
    assert_eq!(sol.status, SolveStatus::Optimal);
    assert!(sol.primal_obj.abs() <= 1e-3, "obj {}", sol.primal_obj);
}

#[test]
fn custom_masses_build_normalized_problems() {
    let g = 4usize;
    let mut src = vec![0.0f64; 16];
    src[5] = 3.0; // single hot cell
    let mut tgt = vec![0.0f64; 16];
    tgt[10] = 7.0;
    tgt[3] = f64::NAN; // junk in ⇒ floored, never poisoning
    let p = transport::problem_from_masses(&src, &tgt, g).unwrap();
    let (rl, ru) = (&p.row_lower, &p.row_upper);
    assert_eq!(rl, ru);
    let s_src: f64 = rl[..16].iter().sum();
    let s_tgt: f64 = rl[16..].iter().sum();
    assert!((s_src - 1.0).abs() < 1e-12 && (s_tgt - 1.0).abs() < 1e-12);
    assert!(rl.iter().all(|&v| v > 0.0 && v.is_finite()));
}

#[test]
fn all_zero_masses_become_uniform() {
    let p = transport::problem_from_masses(&[0.0; 16], &[0.0; 16], 4).unwrap();
    let first = p.row_lower[0];
    assert!(p.row_lower[..16].iter().all(|&v| (v - first).abs() < 1e-15));
    assert!((first - 1.0 / 16.0).abs() < 1e-12);
}

#[test]
fn wrong_length_masses_error() {
    assert!(transport::problem_from_masses(&[1.0; 15], &[1.0; 16], 4).is_err());
}

#[test]
fn custom_solve_matches_preset_solve() {
    use sundial_core::problem::{SolveOptions, SolveStatus};
    let (src, tgt) = transport::masses(transport::Preset::Blobs, 4);
    let p = transport::problem_from_masses(&src, &tgt, 4).unwrap();
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let custom = sundial_core::reference::solve_op(&p, &opts, &mut |_| {});
    let preset = sundial_core::reference::solve_op(
        &transport::problem(transport::Preset::Blobs, 4),
        &opts,
        &mut |_| {},
    );
    assert_eq!(custom.status, SolveStatus::Optimal);
    let rel = (custom.primal_obj - preset.primal_obj).abs() / (1.0 + preset.primal_obj.abs());
    assert!(
        rel <= 1e-6,
        "custom {} vs preset {}",
        custom.primal_obj,
        preset.primal_obj
    );
}
