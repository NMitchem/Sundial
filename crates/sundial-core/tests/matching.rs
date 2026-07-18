//! Point-cloud matching (taxi demo): builder + assignment extraction + integral
//! recovery, validated against brute-force enumeration on tiny instances.
//!
//! Objective scaling: `problem_from_points` scales masses by `1/nt` (H3 — see
//! transport.rs / task-4-report.md), so the LP objective is the coordinate-unit
//! matching cost times `1/nt`. Oracle comparisons therefore use `primal_obj × nt`.
use sundial_core::problem::{ProblemError, SolveOptions, SolveStatus};
use sundial_core::recover::{self, recover_matching};
use sundial_core::reference;
use sundial_core::transport;

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn random_points(n: usize, rng: &mut fastrand::Rng) -> Vec<[f64; 2]> {
    (0..n).map(|_| [rng.f64(), rng.f64()]).collect()
}

fn solve_scaled(riders: &[[f64; 2]], cabs: &[[f64; 2]]) -> sundial_core::problem::Solution {
    let p = transport::problem_from_points(riders, cabs).unwrap();
    let opts = SolveOptions {
        tol: 1e-7,
        max_iters: 2_000_000,
        ..Default::default()
    };
    reference::solve_op(&p, &opts, &mut |_e| {})
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
    // H3 scaling: masses are 1/nt (nt = 4 ⇒ 0.25)
    let mass = 1.0 / 4.0;
    // rider rows: equality at mass; cab rows: [0, mass]
    assert_eq!(&p.row_lower[..3], &[mass, mass, mass]);
    assert_eq!(&p.row_upper[..3], &[mass, mass, mass]);
    assert_eq!(&p.row_lower[3..], &[0.0; 4]);
    assert_eq!(&p.row_upper[3..], &[mass; 4]);
    // c[i*nt+j] is the (unscaled) Euclidean distance
    assert!((p.c[0 * 4 + 2] - dist(riders[0], cabs[2])).abs() < 1e-12);
}

#[test]
fn reference_solve_matches_brute_force() {
    let mut rng = fastrand::Rng::with_seed(7);
    for trial in 0..3 {
        let riders = random_points(4, &mut rng);
        let cabs = random_points(6, &mut rng);
        let nt = cabs.len() as f64;
        let sol = solve_scaled(&riders, &cabs);
        assert_eq!(sol.status, SolveStatus::Optimal, "trial {trial}");
        let bf = brute_force_min_cost(&riders, &cabs);
        // LP obj × nt == matching cost in coordinate units
        let lp_cost = sol.primal_obj * nt;
        assert!(
            (lp_cost - bf).abs() <= 1e-5 * (1.0 + bf.abs()),
            "trial {trial}: lp×nt {lp_cost} vs brute force {bf}",
        );
    }
}

#[test]
fn dominant_assignment_recovers_permutation() {
    let mut rng = fastrand::Rng::with_seed(11);
    let riders = random_points(6, &mut rng);
    let cabs = random_points(8, &mut rng);
    let nt = cabs.len();
    let sol = solve_scaled(&riders, &cabs);
    assert_eq!(sol.status, SolveStatus::Optimal);
    let (assign, min_mass) = transport::dominant_assignment(&sol.x, 6, nt);
    assert_eq!(assign.len(), 6);
    let distinct: std::collections::HashSet<u32> = assign.iter().copied().collect();
    assert_eq!(distinct.len(), 6, "optimal matching must be injective");
    // dominant mass is a fraction of the 1/nt row mass under H3 scaling
    let frac = min_mass * nt as f64;
    assert!(
        frac > 0.9,
        "generic costs ⇒ integral optimum, dominant fraction {frac}"
    );
    // the extracted assignment reproduces the (unscaled) LP objective
    let total: f64 = assign
        .iter()
        .enumerate()
        .map(|(i, &j)| dist(riders[i], cabs[j as usize]))
        .sum();
    let lp_cost = sol.primal_obj * nt as f64;
    assert!(
        (total - lp_cost).abs() <= 1e-4 * (1.0 + lp_cost.abs()),
        "assignment total {total} vs LP×nt {lp_cost}",
    );
}

#[test]
fn dominant_assignment_survives_ties() {
    // one rider exactly between two cabs: the optimal face may be fractional;
    // extraction must stay total (no panic) and report the weak dominance.
    let riders = vec![[0.5, 0.5]];
    let cabs = vec![[0.0, 0.5], [1.0, 0.5]];
    let sol = solve_scaled(&riders, &cabs);
    let (assign, min_mass) = transport::dominant_assignment(&sol.x, 1, 2);
    assert!(assign[0] == 0 || assign[0] == 1);
    // row mass is 1/nt = 0.5; a ~50/50 tie splits it to ~0.25 each
    let frac = min_mass * 2.0;
    assert!(frac > 0.4, "tie may split ~50/50, got fraction {frac}");
}

// ---------------------------------------------------------------------------
// Integral recovery (Task 4a). At n ≤ 7 with nt ≤ 8 the kNN(8) candidate set
// contains every cab, so the graph is complete and recovery is EXACTLY optimal
// — comparable to the brute-force min-cost matching.
// ---------------------------------------------------------------------------

#[test]
fn recovery_matches_brute_force() {
    let mut rng = fastrand::Rng::with_seed(23);
    for trial in 0..5 {
        let riders = random_points(2 + trial, &mut rng); // 2..=6 riders
        let cabs = random_points(8, &mut rng); // 8 cabs ⇒ kNN(8) = complete
        let sol = solve_scaled(&riders, &cabs);
        assert_eq!(sol.status, SolveStatus::Optimal, "trial {trial}");
        let rec = recover_matching(&sol.x, &riders, &cabs);
        let bf = brute_force_min_cost(&riders, &cabs);
        assert!(
            (rec.total_cost - bf).abs() <= 1e-6 * (1.0 + bf.abs()),
            "trial {trial}: recovered {} vs brute force {bf}",
            rec.total_cost,
        );
    }
}

#[test]
fn recovery_is_injective() {
    let mut rng = fastrand::Rng::with_seed(29);
    let riders = random_points(7, &mut rng);
    let cabs = random_points(8, &mut rng);
    let sol = solve_scaled(&riders, &cabs);
    let rec = recover_matching(&sol.x, &riders, &cabs);
    assert_eq!(rec.assignment.len(), riders.len());
    let distinct: std::collections::HashSet<u32> = rec.assignment.iter().copied().collect();
    assert_eq!(distinct.len(), riders.len(), "assignment must be injective");
    for &j in &rec.assignment {
        assert!((j as usize) < cabs.len(), "cab index in range");
    }
    // total_cost is the sum of the actual assigned Euclidean distances
    let recomputed: f64 = rec
        .assignment
        .iter()
        .enumerate()
        .map(|(i, &j)| dist(riders[i], cabs[j as usize]))
        .sum();
    assert!((recomputed - rec.total_cost).abs() <= 1e-12 * (1.0 + recomputed));
}

#[test]
fn recovery_is_crossing_free() {
    let mut rng = fastrand::Rng::with_seed(31);
    let riders = random_points(7, &mut rng);
    let cabs = random_points(8, &mut rng);
    let sol = solve_scaled(&riders, &cabs);
    let rec = recover_matching(&sol.x, &riders, &cabs);
    // exhaustive pair check: no two rider→cab segments properly cross
    let n = riders.len();
    for a in 0..n {
        for b in (a + 1)..n {
            let pa = cabs[rec.assignment[a] as usize];
            let pb = cabs[rec.assignment[b] as usize];
            assert!(
                !recover::segments_cross(riders[a], pa, riders[b], pb),
                "segments {a} and {b} properly cross",
            );
        }
    }
}

#[test]
fn recovery_is_deterministic() {
    let mut rng = fastrand::Rng::with_seed(37);
    let riders = random_points(6, &mut rng);
    let cabs = random_points(8, &mut rng);
    let sol = solve_scaled(&riders, &cabs);
    let a = recover_matching(&sol.x, &riders, &cabs);
    let b = recover_matching(&sol.x, &riders, &cabs);
    assert_eq!(
        a.assignment, b.assignment,
        "assignment must be deterministic"
    );
    assert_eq!(a.total_cost, b.total_cost);
    assert_eq!(a.support_edges, b.support_edges);
}

// ---------------------------------------------------------------------------
// Certified floor (Task 4a fix-round): a rigorous CPU-f64 lower bound on the
// (unscaled, mass-1) matching optimum via a repaired feasible dual. Valid for
// ANY input dual `y` (feasibility is reconstructed), tight when `y` is good.
// Masses are the H3-scaled values (rider_mass = cab_cap = 1/nt).
// ---------------------------------------------------------------------------

#[test]
fn certified_floor_is_valid_lower_bound() {
    let mut rng = fastrand::Rng::with_seed(41);
    for trial in 0..20 {
        let ns = 2 + trial % 5; // 2..=6
        let riders = random_points(ns, &mut rng);
        let cabs = random_points(ns + 2, &mut rng);
        let nt = cabs.len();
        let m = 1.0 / nt as f64;
        let bf = brute_force_min_cost(&riders, &cabs);
        // (i) the solver's dual
        let sol = solve_scaled(&riders, &cabs);
        let f_solved = recover::certified_floor(&sol.y, &riders, &cabs, m, m);
        // (ii) an all-zero dual
        let f_zero = recover::certified_floor(&vec![0.0; ns + nt], &riders, &cabs, m, m);
        // (iii) a garbage dual (feasibility-by-construction must still hold)
        let garbage: Vec<f64> = (0..ns + nt).map(|_| rng.f64() * 20.0 - 10.0).collect();
        let f_garbage = recover::certified_floor(&garbage, &riders, &cabs, m, m);
        for (label, f) in [
            ("solved", f_solved),
            ("zero", f_zero),
            ("garbage", f_garbage),
        ] {
            assert!(
                f <= bf + 1e-9 * (1.0 + bf.abs()),
                "trial {trial} {label}: floor {f} exceeds optimum {bf}",
            );
        }
    }
}

#[test]
fn certified_floor_is_nearly_tight() {
    let mut rng = fastrand::Rng::with_seed(43);
    for trial in 0..5 {
        let riders = random_points(3 + trial, &mut rng); // 3..=7
        let cabs = random_points(8, &mut rng);
        let nt = cabs.len();
        let m = 1.0 / nt as f64;
        let sol = solve_scaled(&riders, &cabs);
        assert_eq!(sol.status, SolveStatus::Optimal, "trial {trial}");
        let bf = brute_force_min_cost(&riders, &cabs);
        let floor = recover::certified_floor(&sol.y, &riders, &cabs, m, m);
        assert!(
            floor <= bf + 1e-9 * (1.0 + bf.abs()),
            "trial {trial}: floor {floor} exceeds optimum {bf}",
        );
        assert!(
            floor >= bf - 1e-3 * (1.0 + bf.abs()),
            "trial {trial}: floor {floor} not tight vs optimum {bf}",
        );
    }
}

#[test]
fn certified_floor_is_deterministic() {
    let mut rng = fastrand::Rng::with_seed(47);
    let riders = random_points(6, &mut rng);
    let cabs = random_points(8, &mut rng);
    let nt = cabs.len();
    let m = 1.0 / nt as f64;
    let sol = solve_scaled(&riders, &cabs);
    let a = recover::certified_floor(&sol.y, &riders, &cabs, m, m);
    let b = recover::certified_floor(&sol.y, &riders, &cabs, m, m);
    assert_eq!(a, b);
}
