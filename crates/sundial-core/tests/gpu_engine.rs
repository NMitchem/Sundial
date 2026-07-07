use sundial_core::gpu::{engine, GpuContext};
use sundial_core::problem::{SolveOptions, SolveStatus};
use sundial_core::{reference, testgen};

#[test]
#[ignore = "requires GPU"]
fn gpu_solves_constructed_lps_to_1e4() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 300_000,
        ..Default::default()
    };
    for seed in 0..4u64 {
        let (p, _, _, obj) = testgen::generate(seed, 40, 25);
        let sol = pollster::block_on(engine::solve_gpu(&ctx, &p, &opts, &mut |_| {})).unwrap();
        assert_eq!(sol.status, SolveStatus::Optimal, "seed {seed}");
        assert!(
            sol.stats.verified.mu() <= 1e-4,
            "seed {seed} mu={}",
            sol.stats.verified.mu()
        );
        let rel = (sol.primal_obj - obj).abs() / (1.0 + obj.abs());
        assert!(rel <= 1e-3, "seed {seed}: obj {} vs {obj}", sol.primal_obj);
    }
}

#[test]
#[ignore = "requires GPU"]
fn gpu_and_reference_agree() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let (p, _, _, _) = testgen::generate(17, 60, 40);
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 300_000,
        ..Default::default()
    };
    let cpu = reference::solve(&p, &opts, &mut |_| {});
    let gpu = pollster::block_on(engine::solve_gpu(&ctx, &p, &opts, &mut |_| {})).unwrap();
    assert_eq!(cpu.status, SolveStatus::Optimal);
    assert_eq!(gpu.status, SolveStatus::Optimal);
    let rel = (cpu.primal_obj - gpu.primal_obj).abs() / (1.0 + cpu.primal_obj.abs());
    assert!(
        rel <= 1e-3,
        "cpu {} vs gpu {}",
        cpu.primal_obj,
        gpu.primal_obj
    );
}
