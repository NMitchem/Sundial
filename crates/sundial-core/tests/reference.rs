use sundial_core::problem::{SolveOptions, SolveStatus};
use sundial_core::{reference, testgen};

#[test]
fn solves_constructed_lps_to_1e6() {
    let opts = SolveOptions {
        tol: 1e-6,
        max_iters: 200_000,
        ..Default::default()
    };
    for seed in 0..5u64 {
        let (p, _x, _y, obj) = testgen::generate(seed, 40, 25);
        let sol = reference::solve(&p, &opts, &mut |_| {});
        assert_eq!(
            sol.status,
            SolveStatus::Optimal,
            "seed {seed} did not converge"
        );
        assert!(
            sol.stats.verified.mu() <= 1e-6,
            "seed {seed} mu={}",
            sol.stats.verified.mu()
        );
        let rel = (sol.primal_obj - obj).abs() / (1.0 + obj.abs());
        assert!(
            rel <= 1e-5,
            "seed {seed}: obj {} vs known {obj}",
            sol.primal_obj
        );
    }
}

#[test]
fn respects_iteration_limit() {
    let (p, _, _, _) = testgen::generate(1, 40, 25);
    let opts = SolveOptions {
        tol: 1e-14,
        max_iters: 100,
        ..Default::default()
    };
    let sol = reference::solve(&p, &opts, &mut |_| {});
    assert_eq!(sol.status, SolveStatus::IterationLimit);
}

#[test]
fn progress_events_fire() {
    let (p, _, _, _) = testgen::generate(2, 40, 25);
    let opts = SolveOptions {
        tol: 1e-6,
        max_iters: 200_000,
        ..Default::default()
    };
    let mut events = 0u32;
    let _ = reference::solve(&p, &opts, &mut |_e| events += 1);
    assert!(events > 0, "no progress events emitted");
}
