use sundial_core::gpu::{buffers, kernels, GpuContext};
use sundial_core::testgen;

fn ctx() -> GpuContext {
    pollster::block_on(GpuContext::new()).expect("no GPU")
}

fn run_once(
    ctx: &GpuContext,
    k: &kernels::Kernels,
    name: &str,
    params: kernels::ParamsData,
    bufs: &[(u32, &wgpu::Buffer)],
    workgroups: u32,
) {
    let pl = k.pipeline(name);
    let ubuf = buffers::uniform_bytes(&ctx.device, &params.bytes(), "params");
    let mut entries: Vec<(u32, &wgpu::Buffer)> = vec![(0, &ubuf)];
    entries.extend_from_slice(bufs);
    let bg = kernels::bind(&ctx.device, pl, &entries);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pl);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);
}

#[test]
#[ignore = "requires GPU"]
fn spmv_matches_cpu() {
    let c = ctx();
    let k = kernels::Kernels::new(&c.device);
    let (p, x, _, _) = testgen::generate(21, 60, 40);
    let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let vals: Vec<f32> = p.a.values.iter().map(|&v| v as f32).collect();
    let b_indptr = buffers::storage_u32(&c.device, &p.a.indptr, "indptr");
    let b_indices = buffers::storage_u32(&c.device, &p.a.indices, "indices");
    let b_vals = buffers::storage_f32(&c.device, &vals, "vals");
    let b_x = buffers::storage_f32(&c.device, &xf, "x");
    let b_out = buffers::storage_zeros_f32(&c.device, p.a.n_rows, "out");
    let params = kernels::ParamsData {
        n: p.a.n_rows as u32,
        stride: 0,
        tau: 0.0,
        sigma: 0.0,
        w: 0.0,
    };
    run_once(
        &c,
        &k,
        "spmv",
        params,
        &[
            (8, &b_indptr),
            (9, &b_indices),
            (1, &b_vals),
            (2, &b_x),
            (6, &b_out),
        ],
        (p.a.n_rows as u32).div_ceil(256),
    );
    let got = pollster::block_on(buffers::readback_f32(
        &c.device, &c.queue, &b_out, p.a.n_rows,
    ));
    let mut want = vec![0.0f64; p.a.n_rows];
    p.a.mul(&x, &mut want);
    for i in 0..want.len() {
        assert!(
            (got[i] as f64 - want[i]).abs() <= 1e-4 * (1.0 + want[i].abs()),
            "row {i}: gpu {} vs cpu {}",
            got[i],
            want[i]
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn primal_step_matches_cpu_with_sentinel_bounds() {
    let c = ctx();
    let k = kernels::Kernels::new(&c.device);
    let n = 1000usize;
    let mut rng = fastrand::Rng::with_seed(5);
    let x: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let aty: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let cv: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    // bounds: mix of finite and sentinel-infinite
    let lv: Vec<f32> = (0..n)
        .map(|i| if i % 3 == 0 { -1e30 } else { -0.5 })
        .collect();
    let uv: Vec<f32> = (0..n)
        .map(|i| if i % 4 == 0 { 1e30 } else { 0.5 })
        .collect();
    let tau = 0.37f32;
    let b = |d: &[f32], l: &str| buffers::storage_f32(&c.device, d, l);
    let (bx, baty, bc, blv, buv) = (
        b(&x, "x"),
        b(&aty, "aty"),
        b(&cv, "c"),
        b(&lv, "lv"),
        b(&uv, "uv"),
    );
    let bxn = buffers::storage_zeros_f32(&c.device, n, "x_new");
    let bxt = buffers::storage_zeros_f32(&c.device, n, "x_tilde");
    let params = kernels::ParamsData {
        n: n as u32,
        stride: 0,
        tau,
        sigma: 0.0,
        w: 0.0,
    };
    run_once(
        &c,
        &k,
        "primal_step",
        params,
        &[
            (1, &bx),
            (2, &baty),
            (3, &bc),
            (4, &blv),
            (5, &buv),
            (6, &bxn),
            (7, &bxt),
        ],
        (n as u32).div_ceil(256),
    );
    let got_xn = pollster::block_on(buffers::readback_f32(&c.device, &c.queue, &bxn, n));
    let got_xt = pollster::block_on(buffers::readback_f32(&c.device, &c.queue, &bxt, n));
    for j in 0..n {
        let v = x[j] - tau * (cv[j] + aty[j]);
        let want = v.clamp(lv[j], uv[j]);
        assert!(
            (got_xn[j] - want).abs() <= 1e-6 * (1.0 + want.abs()),
            "j={j}"
        );
        assert!((got_xt[j] - (2.0 * want - x[j])).abs() <= 1e-6, "j={j}");
    }
}

#[test]
#[ignore = "requires GPU"]
fn reduce_dot_one_million_matches_f64() {
    let c = ctx();
    let k = kernels::Kernels::new(&c.device);
    let n = 1_000_000usize;
    let mut rng = fastrand::Rng::with_seed(9);
    let a: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.f32() * 2.0 - 1.0).collect();
    let ba = buffers::storage_f32(&c.device, &a, "a");
    let bb = buffers::storage_f32(&c.device, &b, "b");
    let red = kernels::Reducer::new(&c.device, n);
    let got = pollster::block_on(red.dot(&c, &k, &ba, &bb, n)) as f64;
    let want: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
    assert!(
        (got - want).abs() <= 1e-3 * (1.0 + want.abs()),
        "dot: gpu {got} vs f64 {want}"
    );
}
