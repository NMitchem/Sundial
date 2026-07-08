use sundial_core::gpu::op::{GpuOp, TransportGpuOp};
use sundial_core::gpu::{buffers, kernels::Kernels, GpuContext};
use sundial_core::linop::LinOp;
use sundial_core::problem::{SolveOptions, SolveStatus};
use sundial_core::transport::{self, Preset, TransportOp};

fn assert_close(gpu: &[f32], cpu: &[f64], what: &str) {
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        assert!(
            ((*g as f64) - c).abs() <= 1e-4 * (1.0 + c.abs()),
            "{what}[{i}]: gpu {g} vs cpu {c}"
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn transport_kernels_match_cpu_op() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let k = Kernels::new(&ctx.device);
    let (ns, nt) = (256usize, 256usize); // g=16 → n = 65_536
    let (n, m) = (ns * nt, ns + nt);
    let op = TransportOp { ns, nt };
    let gop = TransportGpuOp::new(&ctx.device, ns, nt);
    let mut rng = fastrand::Rng::with_seed(23);
    let x: Vec<f64> = (0..n).map(|_| rng.f64() - 0.5).collect();
    let y: Vec<f64> = (0..m).map(|_| rng.f64() - 0.5).collect();

    let xb = buffers::storage_f32(
        &ctx.device,
        &x.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        "x",
    );
    let yb = buffers::storage_f32(
        &ctx.device,
        &y.iter().map(|&v| v as f32).collect::<Vec<_>>(),
        "y",
    );
    let ax = buffers::storage_zeros_f32(&ctx.device, m, "ax");
    let aty = buffers::storage_zeros_f32(&ctx.device, n, "aty");

    let mut enc = ctx.device.create_command_encoder(&Default::default());
    gop.record_apply(&ctx.device, &k, &mut enc, &xb, &ax);
    gop.record_apply_t(&ctx.device, &k, &mut enc, &yb, &aty);
    ctx.queue.submit([enc.finish()]);

    let (mut ax_cpu, mut aty_cpu) = (vec![0.0; m], vec![0.0; n]);
    op.apply(&x, &mut ax_cpu);
    op.apply_t(&y, &mut aty_cpu);
    let ax_gpu = pollster::block_on(buffers::readback_f32(&ctx.device, &ctx.queue, &ax, m));
    let aty_gpu = pollster::block_on(buffers::readback_f32(&ctx.device, &ctx.queue, &aty, n));
    assert_close(&ax_gpu, &ax_cpu, "ax");
    assert_close(&aty_gpu, &aty_cpu, "aty");
}

#[test]
#[ignore = "requires GPU"]
fn gpu_solves_small_transport() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let g = 8usize; // n = 4096, m = 128
    let p = transport::problem(Preset::Blobs, g);
    let gop = TransportGpuOp::new(&ctx.device, g * g, g * g);
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = pollster::block_on(sundial_core::gpu::engine::solve_gpu_op(
        &ctx,
        &p,
        &gop,
        &opts,
        &mut |_| {},
        None,
    ))
    .unwrap();
    assert_eq!(sol.status, SolveStatus::Optimal);
    assert!(
        sol.stats.verified.mu() <= 1e-4,
        "mu={}",
        sol.stats.verified.mu()
    );
    // cross-check against the CPU reference on the same problem
    let cpu = sundial_core::reference::solve_op(&p, &opts, &mut |_| {});
    let rel = (sol.primal_obj - cpu.primal_obj).abs() / (1.0 + cpu.primal_obj.abs());
    assert!(
        rel <= 1e-3,
        "gpu {} vs cpu {}",
        sol.primal_obj,
        cpu.primal_obj
    );
}

#[test]
#[ignore = "requires GPU"]
fn gpu_transport_1m_variables_to_1e4() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let g = 32usize;
    let p = transport::problem(Preset::Blobs, g);
    assert_eq!(p.n_vars(), 1_048_576, "hero must be ≥1M variables");
    let gop = TransportGpuOp::new(&ctx.device, g * g, g * g);
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = pollster::block_on(sundial_core::gpu::engine::solve_gpu_op(
        &ctx,
        &p,
        &gop,
        &opts,
        &mut |_| {},
        None,
    ))
    .unwrap();
    eprintln!(
        "1M transport: {:?}, {} iters, {} restarts, {:.0} ms, verified mu {:.2e}",
        sol.status, sol.stats.iterations, sol.stats.restarts, sol.stats.solve_ms,
        sol.stats.verified.mu()
    );
    assert_eq!(sol.status, SolveStatus::Optimal);
    assert!(sol.stats.verified.mu() <= 1e-4, "mu={}", sol.stats.verified.mu());
}
