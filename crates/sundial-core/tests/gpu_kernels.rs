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

// Projected-gap guards: a dual-objective term whose sign-relevant bound is the
// open ±1e30 sentinel must contribute EXACTLY 0 (never leak 1e30 × dual), while
// the dual residual `rd` still reports the infeasible component unchanged.
#[test]
#[ignore = "requires GPU"]
fn residual_terms_project_open_bounds_to_zero() {
    let c = ctx();
    let k = kernels::Kernels::new(&c.device);
    let inf = 1e30f32;
    let params = || kernels::ParamsData {
        n: 4,
        stride: 0,
        tau: 0.0,
        sigma: 0.0,
        w: 0.0,
    };
    let b = |d: &[f32], l: &str| buffers::storage_f32(&c.device, d, l);

    // dual_res_terms: in_a=aty in_b=c in_c=lv in_d=uv out_a=rd out_b=bterm
    // j0: g>0, finite lower  -> absorbed (rd=0), bterm=g*lv
    // j1: g>0, OPEN lower     -> rd=g, bterm=0 (guarded)
    // j2: g<0, finite upper  -> absorbed (rd=0), bterm=g*uv
    // j3: g<0, OPEN upper     -> rd=g, bterm=0 (guarded)
    let aty = [0.0f32; 4];
    let cv = [1.0f32, 2.0, -1.5, -3.0];
    let lv = [0.5f32, -inf, -inf, -inf];
    let uv = [inf, inf, 0.8, inf];
    let exp_rd = [0.0f32, 2.0, 0.0, -3.0];
    let exp_bt = [0.5f32, 0.0, -1.2, 0.0];
    let (b_aty, b_c, b_lv, b_uv) = (b(&aty, "aty"), b(&cv, "c"), b(&lv, "lv"), b(&uv, "uv"));
    let b_rd = buffers::storage_zeros_f32(&c.device, 4, "rd");
    let b_bt = buffers::storage_zeros_f32(&c.device, 4, "bterm");
    run_once(
        &c,
        &k,
        "dual_res_terms",
        params(),
        &[
            (1, &b_aty),
            (2, &b_c),
            (3, &b_lv),
            (4, &b_uv),
            (6, &b_rd),
            (7, &b_bt),
        ],
        1,
    );
    let got_rd = pollster::block_on(buffers::readback_f32(&c.device, &c.queue, &b_rd, 4));
    let got_bt = pollster::block_on(buffers::readback_f32(&c.device, &c.queue, &b_bt, 4));
    for j in 0..4 {
        assert!(
            (got_rd[j] - exp_rd[j]).abs() <= 1e-6,
            "rd[{j}]={} want {}",
            got_rd[j],
            exp_rd[j]
        );
        assert!(
            (got_bt[j] - exp_bt[j]).abs() <= 1e-4,
            "bterm[{j}]={} want {}",
            got_bt[j],
            exp_bt[j]
        );
    }
    // open-bound columns: exactly 0 (no sentinel leak), and rd == g unchanged
    assert_eq!(got_bt[1], 0.0);
    assert_eq!(got_bt[3], 0.0);
    assert_eq!(got_rd[1], cv[1] + aty[1]);
    assert_eq!(got_rd[3], cv[3] + aty[3]);

    // row_terms: in_a=y in_b=lc in_c=uc out_a=rterm
    // i0: y>0, finite upper -> uc*y ; i1: y>0, OPEN upper -> 0 (guarded)
    // i2: y<0, finite lower -> lc*y ; i3: y<0, OPEN lower -> 0 (guarded)
    let y = [0.5f32, 0.5, -0.4, -0.3];
    let lc = [-inf, -3.0, -1.0, -inf];
    let uc = [2.0f32, inf, inf, 5.0];
    let exp_rt = [1.0f32, 0.0, 0.4, 0.0];
    let (b_y, b_lc, b_uc) = (b(&y, "y"), b(&lc, "lc"), b(&uc, "uc"));
    let b_rt = buffers::storage_zeros_f32(&c.device, 4, "rterm");
    run_once(
        &c,
        &k,
        "row_terms",
        params(),
        &[(1, &b_y), (2, &b_lc), (3, &b_uc), (6, &b_rt)],
        1,
    );
    let got_rt = pollster::block_on(buffers::readback_f32(&c.device, &c.queue, &b_rt, 4));
    for i in 0..4 {
        assert!(
            (got_rt[i] - exp_rt[i]).abs() <= 1e-6,
            "rterm[{i}]={} want {}",
            got_rt[i],
            exp_rt[i]
        );
    }
    assert_eq!(got_rt[1], 0.0);
    assert_eq!(got_rt[3], 0.0);
}
