//! Point-cloud matching (taxi demo): builder + assignment extraction,
//! validated against brute-force enumeration on tiny instances.
use sundial_core::problem::{ProblemError, SolveOptions, SolveStatus};
use sundial_core::reference;
use sundial_core::transport;

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn random_points(n: usize, rng: &mut fastrand::Rng) -> Vec<[f64; 2]> {
    (0..n).map(|_| [rng.f64(), rng.f64()]).collect()
}

/// Minimum-cost injective rider→cab assignment by exhaustive recursion.
/// Tiny sizes only (O(nt!/(nt-ns)!)).
fn brute_force_min_cost(riders: &[[f64; 2]], cabs: &[[f64; 2]]) -> f64 {
    fn rec(
        i: usize,
        riders: &[[f64; 2]],
        cabs: &[[f64; 2]],
        used: &mut [bool],
        acc: f64,
        best: &mut f64,
    ) {
        if i == riders.len() {
            *best = best.min(acc);
            return;
        }
        for j in 0..cabs.len() {
            if used[j] {
                continue;
            }
            used[j] = true;
            rec(
                i + 1,
                riders,
                cabs,
                used,
                acc + dist(riders[i], cabs[j]),
                best,
            );
            used[j] = false;
        }
    }
    let mut best = f64::INFINITY;
    rec(
        0,
        riders,
        cabs,
        &mut vec![false; cabs.len()],
        0.0,
        &mut best,
    );
    best
}

#[test]
fn builder_rejects_bad_inputs() {
    let p2 = vec![[0.0, 0.0], [1.0, 1.0]];
    let p1 = vec![[0.5, 0.5]];
    // more riders than cabs → capacity-infeasible by construction
    assert!(matches!(
        transport::problem_from_points(&p2, &p1),
        Err(ProblemError::Dimension(_))
    ));
    // empty riders
    assert!(matches!(
        transport::problem_from_points(&[], &p1),
        Err(ProblemError::Dimension(_))
    ));
    // non-finite coordinate
    let bad = vec![[f64::NAN, 0.0]];
    assert!(matches!(
        transport::problem_from_points(&bad, &p1),
        Err(ProblemError::Dimension(_))
    ));
}

#[test]
#[allow(clippy::erasing_op, clippy::identity_op)] // 0 * 4 + 2 spells out i*nt+j
fn builder_shapes_and_bounds() {
    let riders = vec![[0.1, 0.2], [0.8, 0.9], [0.4, 0.4]];
    let cabs = vec![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.2, 0.9]];
    let p = transport::problem_from_points(&riders, &cabs).unwrap();
    assert_eq!(p.n_vars(), 12);
    assert_eq!(p.n_cons(), 7);
    // rider rows: equality at 1; cab rows: [0, 1]
    assert_eq!(&p.row_lower[..3], &[1.0, 1.0, 1.0]);
    assert_eq!(&p.row_upper[..3], &[1.0, 1.0, 1.0]);
    assert_eq!(&p.row_lower[3..], &[0.0; 4]);
    assert_eq!(&p.row_upper[3..], &[1.0; 4]);
    // c[i*nt+j] is the Euclidean distance
    assert!((p.c[0 * 4 + 2] - dist(riders[0], cabs[2])).abs() < 1e-12);
}

#[test]
fn reference_solve_matches_brute_force() {
    let mut rng = fastrand::Rng::with_seed(7);
    for trial in 0..3 {
        let riders = random_points(4, &mut rng);
        let cabs = random_points(6, &mut rng);
        let p = transport::problem_from_points(&riders, &cabs).unwrap();
        let opts = SolveOptions {
            tol: 1e-7,
            max_iters: 2_000_000,
            ..Default::default()
        };
        let sol = reference::solve_op(&p, &opts, &mut |_e| {});
        assert_eq!(sol.status, SolveStatus::Optimal, "trial {trial}");
        let bf = brute_force_min_cost(&riders, &cabs);
        assert!(
            (sol.primal_obj - bf).abs() <= 1e-5 * (1.0 + bf.abs()),
            "trial {trial}: lp {} vs brute force {bf}",
            sol.primal_obj
        );
    }
}

#[test]
fn dominant_assignment_recovers_permutation() {
    let mut rng = fastrand::Rng::with_seed(11);
    let riders = random_points(6, &mut rng);
    let cabs = random_points(8, &mut rng);
    let p = transport::problem_from_points(&riders, &cabs).unwrap();
    let opts = SolveOptions {
        tol: 1e-7,
        max_iters: 2_000_000,
        ..Default::default()
    };
    let sol = reference::solve_op(&p, &opts, &mut |_e| {});
    assert_eq!(sol.status, SolveStatus::Optimal);
    let (assign, min_mass) = transport::dominant_assignment(&sol.x, 6, 8);
    assert_eq!(assign.len(), 6);
    let distinct: std::collections::HashSet<u32> = assign.iter().copied().collect();
    assert_eq!(distinct.len(), 6, "optimal matching must be injective");
    assert!(
        min_mass > 0.9,
        "generic costs ⇒ integral optimum, got {min_mass}"
    );
    // the extracted assignment reproduces the LP objective
    let total: f64 = assign
        .iter()
        .enumerate()
        .map(|(i, &j)| dist(riders[i], cabs[j as usize]))
        .sum();
    assert!(
        (total - sol.primal_obj).abs() <= 1e-4 * (1.0 + sol.primal_obj.abs()),
        "assignment total {total} vs objective {}",
        sol.primal_obj
    );
}

#[test]
fn dominant_assignment_survives_ties() {
    // one rider exactly between two cabs: the optimal face may be fractional;
    // extraction must stay total (no panic) and report the weak dominance.
    let riders = vec![[0.5, 0.5]];
    let cabs = vec![[0.0, 0.5], [1.0, 0.5]];
    let p = transport::problem_from_points(&riders, &cabs).unwrap();
    let opts = SolveOptions {
        tol: 1e-7,
        max_iters: 2_000_000,
        ..Default::default()
    };
    let sol = reference::solve_op(&p, &opts, &mut |_e| {});
    let (assign, min_mass) = transport::dominant_assignment(&sol.x, 1, 2);
    assert!(assign[0] == 0 || assign[0] == 1);
    assert!(min_mass > 0.4, "tie may split ~50/50, got {min_mass}");
}
