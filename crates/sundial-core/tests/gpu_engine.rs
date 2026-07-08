use sundial_core::gpu::op::CsrGpuOp;
use sundial_core::gpu::{engine, GpuContext};
use sundial_core::linop::LinOp;
use sundial_core::problem::{OpProblem, SolveOptions, SolveStatus};
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
#[ignore = "requires GPU"]
fn solve_gpu_op_unscaled_csr_reaches_optimal() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let (p, _, _, obj) = testgen::generate(2, 30, 20);
    let opp = OpProblem::new(
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
    let gop = CsrGpuOp::new(&ctx.device, &p.a, &p.at);
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let mut snaps = 0usize;
    let mut snap_len = 0usize;
    let sol = pollster::block_on(engine::solve_gpu_op(
        &ctx,
        &opp,
        &gop,
        &opts,
        &mut |_| {},
        Some(&mut |s: sundial_core::problem::SnapshotEvent| {
            snaps += 1;
            snap_len = s.ax.len();
        }),
    ))
    .unwrap();
    assert_eq!(sol.status, SolveStatus::Optimal);
    assert!(
        sol.stats.verified.mu() <= 1e-4,
        "mu={}",
        sol.stats.verified.mu()
    );
    let rel = (sol.primal_obj - obj).abs() / (1.0 + obj.abs());
    assert!(rel <= 1e-3, "obj {} vs {obj}", sol.primal_obj);
    assert!(snaps >= 1, "snapshot callback never fired");
    assert_eq!(snap_len, p.n_cons(), "snapshot must carry A·x (m floats)");
}
