use super::buffers::{self, pack_f32_inf_sentinel};
use super::kernels::{self, Kernels, ParamsData, Reducer};
use super::op;
use super::{wgs, GpuContext, GpuError, WG};
use crate::problem::*;
use crate::{kkt, reference, scale};
use web_time::Instant;

struct EvalOut {
    rel_p: f64,
    rel_d: f64,
    rel_gap: f64,
    mu: f64,
}

/// Explicit-path entry: Ruiz/PC-scaled CSR solved on the GPU, certificate
/// verified in ORIGINAL space. Signature is UNCHANGED (wasm.rs + the CLI
/// call this).
pub async fn solve_gpu(
    ctx: &GpuContext,
    p: &LpProblem,
    opts: &SolveOptions,
    progress: &mut dyn FnMut(ProgressEvent),
) -> Result<Solution, GpuError> {
    assert!(
        opts.check_every > 0,
        "SolveOptions::check_every must be > 0"
    );
    let (sp, s) = scale::ruiz_pc(p, 10);

    // buffer-size guard: largest single binding is the CSR values/indices array
    let needed_mib =
        ((sp.a.nnz() * 4).max((sp.n_vars().max(sp.n_cons())) * 4) as u64).div_ceil(1024 * 1024);
    if needed_mib > ctx.max_binding_mib {
        return Err(GpuError::BufferTooLarge {
            needed_mib,
            allowed_mib: ctx.max_binding_mib,
        });
    }

    let norm_a = reference::power_iteration_norm(&sp.a, &sp.at, 100, opts.seed);
    let gop = op::CsrGpuOp::new(&ctx.device, &sp.a, &sp.at);
    solve_core(
        ctx,
        &sp.view(),
        &p.view(),
        &s,
        norm_a,
        &gop,
        opts,
        progress,
        None,
    )
    .await
}

/// Operator (matrix-free) entry: the constraint matrix is recorded by
/// `gpu_op`, never materialized. Matrix-free problems solve UNSCALED with the
/// operator's exact norm; certificate space == iterate space.
pub async fn solve_gpu_op<O: crate::linop::LinOp>(
    ctx: &GpuContext,
    p: &OpProblem<O>,
    gpu_op: &dyn op::GpuOp,
    opts: &SolveOptions,
    progress: &mut dyn FnMut(ProgressEvent),
    snapshot: Option<&mut dyn FnMut(SnapshotEvent<'_>)>,
) -> Result<Solution, GpuError> {
    assert!(
        opts.check_every > 0,
        "SolveOptions::check_every must be > 0"
    );
    assert_eq!(gpu_op.n_rows(), p.n_cons(), "GpuOp/OpProblem row mismatch");
    assert_eq!(gpu_op.n_cols(), p.n_vars(), "GpuOp/OpProblem col mismatch");
    let needed_mib = (((p.n_vars().max(p.n_cons())) * 4).div_ceil(1024 * 1024)) as u64;
    if needed_mib > ctx.max_binding_mib {
        return Err(GpuError::BufferTooLarge {
            needed_mib,
            allowed_mib: ctx.max_binding_mib,
        });
    }
    // Matrix-free = UNSCALED with the operator's exact norm (adjudication).
    let norm_a = crate::linop::op_norm2(&p.op, opts.seed);
    let s = scale::Scaling::identity(p.n_cons(), p.n_vars());
    let v = p.view();
    solve_core(ctx, &v, &v, &s, norm_a, gpu_op, opts, progress, snapshot).await
}

#[allow(clippy::too_many_arguments)]
async fn solve_core(
    ctx: &GpuContext,
    it: &LpView<'_>,   // iterate space (scaled explicit / original op)
    orig: &LpView<'_>, // certificate space — CPU f64 checks happen here
    s: &scale::Scaling,
    norm_a: f64,
    gpu_op: &dyn op::GpuOp,
    opts: &SolveOptions,
    progress: &mut dyn FnMut(ProgressEvent),
    mut snapshot: Option<&mut dyn FnMut(SnapshotEvent<'_>)>,
) -> Result<Solution, GpuError> {
    let start = Instant::now();
    let (m, n) = (gpu_op.n_rows(), gpu_op.n_cols());
    let (tau, sigma) = ((0.9 / norm_a) as f32, (0.9 / norm_a) as f32);
    let (q_norm, c_norm) = kkt::denominators_view(orig); // ORIGINAL-space denominators, f64
    let dev = &ctx.device;
    let k = Kernels::new(dev);
    let red = Reducer::new(dev, n.max(m));
    // results[0]=maxabs, [1..=5]=cur eval, [6..=10]=avg eval, [11..16] spare,
    // [16..16+m]=ORIGINAL-space A·x_cur snapshot (written only when requested).
    let results = buffers::storage_zeros_f32(dev, 16 + m, "results");
    let has_snapshot = snapshot.is_some();

    // ---- pack + upload (the six CSR buffers now live in `gpu_op`). ----
    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let b_f = |d: &[f32], l: &str| buffers::storage_f32(dev, d, l);

    // iterate-space data (sentinel-packed bounds)
    let c_s = b_f(&f32v(it.c), "c_s");
    let lv_s = b_f(&pack_f32_inf_sentinel(it.col_lower), "lv_s");
    let uv_s = b_f(&pack_f32_inf_sentinel(it.col_upper), "uv_s");
    let lc_s = b_f(&pack_f32_inf_sentinel(it.row_lower), "lc_s");
    let uc_s = b_f(&pack_f32_inf_sentinel(it.row_upper), "uc_s");

    // ORIGINAL-space data for residual evaluation. Bounds use the SAME 1e30
    // sentinel as the iterate bounds (no true ±∞ in GPU buffers — WGSL may assume
    // finite math, so ±∞ is not portable across browser backends). The residual
    // kernels never multiply a non-finite bound: `dual_res_terms`/`row_terms` guard
    // every bound term by its INF_THRESH finiteness flag, computing the gap at the
    // sign-cone projection of the dual — identical semantics to the host
    // `project_dual` f64 gate, so the GPU trigger and the CPU verdict agree and
    // `rel_gap` stays finite for progress events.
    let c_o = b_f(&f32v(orig.c), "c_o");
    let lv_o = b_f(&pack_f32_inf_sentinel(orig.col_lower), "lv_o");
    let uv_o = b_f(&pack_f32_inf_sentinel(orig.col_upper), "uv_o");
    let lc_o = b_f(&pack_f32_inf_sentinel(orig.row_lower), "lc_o");
    let uc_o = b_f(&pack_f32_inf_sentinel(orig.row_upper), "uc_o");
    let dr = b_f(&f32v(&s.row), "dr");
    let dr_inv = b_f(
        &f32v(&s.row.iter().map(|v| 1.0 / v).collect::<Vec<_>>()),
        "dr_inv",
    );
    let dc_inv = b_f(
        &f32v(&s.col.iter().map(|v| 1.0 / v).collect::<Vec<_>>()),
        "dc_inv",
    );

    // iterates (x0 = clamp(0, bounds), y0 = 0), sums = copies of the start point
    let x0: Vec<f32> = it
        .col_lower
        .iter()
        .zip(it.col_upper)
        .map(|(&l, &u)| (0.0f64.clamp(l, u)) as f32)
        .collect();
    let x = b_f(&x0, "x");
    let x_new = b_f(&x0, "x_new");
    let x_tilde = buffers::storage_zeros_f32(dev, n, "x_tilde");
    let y = buffers::storage_zeros_f32(dev, m, "y");
    let y_new = buffers::storage_zeros_f32(dev, m, "y_new");
    let sum_x = b_f(&x0, "sum_x");
    let sum_y = buffers::storage_zeros_f32(dev, m, "sum_y");

    // scratch for iteration + residual evaluation
    let aty = buffers::storage_zeros_f32(dev, n, "aty");
    let axt = buffers::storage_zeros_f32(dev, m, "axt");
    let ax_s = buffers::storage_zeros_f32(dev, m, "ax_s");
    let ax_o = buffers::storage_zeros_f32(dev, m, "ax_o");
    let aty_s = buffers::storage_zeros_f32(dev, n, "aty_s");
    let aty_o = buffers::storage_zeros_f32(dev, n, "aty_o");
    let rp = buffers::storage_zeros_f32(dev, m, "rp");
    let rd = buffers::storage_zeros_f32(dev, n, "rd");
    let bterm = buffers::storage_zeros_f32(dev, n, "bterm");
    let rterm = buffers::storage_zeros_f32(dev, m, "rterm");
    let y_o = buffers::storage_zeros_f32(dev, m, "y_o");
    let xa = buffers::storage_zeros_f32(dev, n, "xa");
    let ya = buffers::storage_zeros_f32(dev, m, "ya");

    // ---- static uniforms ----
    let u = |n_: usize, w: f32, label: &str| {
        buffers::uniform_bytes(
            dev,
            &ParamsData {
                n: n_ as u32,
                stride: wgs(n_) * WG,
                tau,
                sigma,
                w,
            }
            .bytes(),
            label,
        )
    };
    let u_n = u(n, 0.0, "u_n");
    let u_m = u(m, 0.0, "u_m");

    // ---- iteration bind groups, per parity ----
    // Only the four vector kernels (primal_step, dual_step, accum×2) are
    // prebuilt; the op records the two spmv steps per iteration.
    // parity 0 reads (x, y) writes (x_new, y_new); parity 1 the reverse.
    let make_iter =
        |x_src: &wgpu::Buffer, y_src: &wgpu::Buffer, x_dst: &wgpu::Buffer, y_dst: &wgpu::Buffer| {
            [
                // 1. x_dst, x_tilde = primal_step(x_src, aty)
                (
                    k.pipeline("primal_step"),
                    kernels::bind(
                        dev,
                        k.pipeline("primal_step"),
                        &[
                            (0, &u_n),
                            (1, x_src),
                            (2, &aty),
                            (3, &c_s),
                            (4, &lv_s),
                            (5, &uv_s),
                            (6, x_dst),
                            (7, &x_tilde),
                        ],
                    ),
                    wgs(n),
                ),
                // 2. y_dst = dual_step(y_src, axt)
                (
                    k.pipeline("dual_step"),
                    kernels::bind(
                        dev,
                        k.pipeline("dual_step"),
                        &[
                            (0, &u_m),
                            (1, y_src),
                            (2, &axt),
                            (3, &lc_s),
                            (4, &uc_s),
                            (6, y_dst),
                        ],
                    ),
                    wgs(m),
                ),
                // 3. sum_x += x_dst
                (
                    k.pipeline("accum"),
                    kernels::bind(
                        dev,
                        k.pipeline("accum"),
                        &[(0, &u_n), (1, x_dst), (6, &sum_x)],
                    ),
                    wgs(n),
                ),
                // 4. sum_y += y_dst
                (
                    k.pipeline("accum"),
                    kernels::bind(
                        dev,
                        k.pipeline("accum"),
                        &[(0, &u_m), (1, y_dst), (6, &sum_y)],
                    ),
                    wgs(m),
                ),
            ]
        };
    let iter_bg = [
        make_iter(&x, &y, &x_new, &y_new),
        make_iter(&x_new, &y_new, &x, &y),
    ];
    // Per-parity iterate handles for the op's spmv sources: (y_src, x̃).
    let parity_src: [(&wgpu::Buffer, &wgpu::Buffer); 2] = [(&y, &x_tilde), (&y_new, &x_tilde)];

    let bufs = EvalBufs {
        c_s: &c_s,
        c_o: &c_o,
        lv_o: &lv_o,
        uv_o: &uv_o,
        lc_o: &lc_o,
        uc_o: &uc_o,
        dr: &dr,
        dr_inv: &dr_inv,
        dc_inv: &dc_inv,
        ax_s: &ax_s,
        ax_o: &ax_o,
        aty_s: &aty_s,
        aty_o: &aty_o,
        rp: &rp,
        rd: &rd,
        bterm: &bterm,
        rterm: &rterm,
        y_o: &y_o,
    };

    // ---- main loop ----
    let mut iter: u64 = 0;
    let mut parity: usize = 0; // which bind-group set to use for the NEXT iteration
    let mut count: u64 = 1; // number of points folded into the running sums
    let mut mu_last_restart = f64::INFINITY;
    let mut iters_since_restart: u64 = 0;
    let mut restarts: u32 = 0;
    let mut trigger_tol = opts.tol;
    let mut status = SolveStatus::IterationLimit;
    let mut last_check_time = Instant::now();
    let mut last_check_iter: u64 = 0;

    'outer: while iter < opts.max_iters {
        // one submit = check_every iterations
        let mut enc = dev.create_command_encoder(&Default::default());
        for _ in 0..opts.check_every {
            let (y_src, xt) = parity_src[parity];
            let steps = &iter_bg[parity]; // [primal_step, dual_step, accum_x, accum_y]
            gpu_op.record_apply_t(dev, &k, &mut enc, y_src, &aty); // aty = Aᵀ y_src
            dispatch_prebuilt(&mut enc, &steps[0]); // primal_step → x_dst, x̃
            gpu_op.record_apply(dev, &k, &mut enc, xt, &axt); // axt = A x̃
            dispatch_prebuilt(&mut enc, &steps[1]); // dual_step → y_dst
            dispatch_prebuilt(&mut enc, &steps[2]); // sum_x += x_dst
            dispatch_prebuilt(&mut enc, &steps[3]); // sum_y += y_dst
            parity ^= 1;
            iter += 1;
            count += 1;
        }
        ctx.queue.submit([enc.finish()]);
        // current iterate now lives in the buffers the NEXT parity reads from
        let (x_cur, y_cur): (&wgpu::Buffer, &wgpu::Buffer) = if parity == 0 {
            (&x, &y)
        } else {
            (&x_new, &y_new)
        };

        // ---- one encoder per check: averages + watchdog + both evals ----
        let mut enc = dev.create_command_encoder(&Default::default());
        {
            let u_avg_n = buffers::uniform_bytes(
                dev,
                &ParamsData {
                    n: n as u32,
                    stride: wgs(n) * WG,
                    tau,
                    sigma,
                    w: 1.0 / count as f32,
                }
                .bytes(),
                "u_avg_n",
            );
            let u_avg_m = buffers::uniform_bytes(
                dev,
                &ParamsData {
                    n: m as u32,
                    stride: wgs(m) * WG,
                    tau,
                    sigma,
                    w: 1.0 / count as f32,
                }
                .bytes(),
                "u_avg_m",
            );
            for (ub, src, dst, len) in [(&u_avg_n, &sum_x, &xa, n), (&u_avg_m, &sum_y, &ya, m)] {
                let pl = k.pipeline("ew_scale");
                let bg = kernels::bind(dev, pl, &[(0, ub), (1, src), (6, dst)]);
                kernels::pass_dispatch(&mut enc, pl, &bg, wgs(len));
            }
        }
        red.record_maxabs(dev, &mut enc, &k, x_cur, n, &results, 0);
        record_eval(
            dev, &mut enc, &k, &red, gpu_op, &u_n, &u_m, &bufs, x_cur, y_cur, n, m, &results, 1,
        );
        if has_snapshot {
            // ax_o now holds ORIGINAL-space A·x_cur from the CURRENT eval; copy
            // it out BEFORE the average's record_eval below overwrites ax_o.
            enc.copy_buffer_to_buffer(&ax_o, 0, &results, 16 * 4, (m * 4) as u64);
        }
        record_eval(
            dev, &mut enc, &k, &red, gpu_op, &u_n, &u_m, &bufs, &xa, &ya, n, m, &results, 6,
        );
        ctx.queue.submit([enc.finish()]);
        let want = if has_snapshot { 16 + m } else { 11 };
        let vals = buffers::readback_f32(dev, &ctx.queue, &results, want).await;

        // NaN/overflow watchdog (evaluations for a broken iterate are unused)
        let mx = vals[0];
        if !mx.is_finite() || mx > 1e29 {
            status = SolveStatus::NumericalBreakdown;
            break 'outer;
        }
        let e_cur = eval_from(&vals, 1, orig.obj_offset, q_norm, c_norm);
        let e_avg = eval_from(&vals, 6, orig.obj_offset, q_norm, c_norm);
        let (e_cand, cand_x, cand_y, cand_is_avg) = if e_avg.mu < e_cur.mu {
            (&e_avg, &xa, &ya, true)
        } else {
            (&e_cur, x_cur, y_cur, false)
        };

        let now = Instant::now();
        let ms_per_iter = now.duration_since(last_check_time).as_secs_f64() * 1000.0
            / (iter - last_check_iter).max(1) as f64;
        last_check_time = now;
        last_check_iter = iter;
        progress(ProgressEvent {
            iter,
            rel_primal: e_cur.rel_p,
            rel_dual: e_cur.rel_d,
            rel_gap: e_cur.rel_gap,
            ms_per_iter,
        });

        if let Some(cb) = snapshot.as_deref_mut() {
            cb(SnapshotEvent {
                iter,
                ax: &vals[16..16 + m],
            });
        }

        // termination trigger → authoritative CPU f64 verification (ORIGINAL space)
        if e_cand.mu <= trigger_tol {
            let xs = buffers::readback_f32(dev, &ctx.queue, cand_x, n).await;
            let ys = buffers::readback_f32(dev, &ctx.queue, cand_y, m).await;
            let xs64: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
            let ys64: Vec<f64> = ys.iter().map(|&v| v as f64).collect();
            let xo = s.unscale_x(&xs64);
            let mut yo = s.unscale_y(&ys64);
            project_dual(orig.row_lower, orig.row_upper, &mut yo);
            let verified = kkt::residuals_view(orig, &xo, &yo);
            if verified.mu() <= opts.tol {
                return Ok(finish(
                    xo,
                    yo,
                    verified,
                    SolveStatus::Optimal,
                    iter,
                    restarts,
                    start,
                ));
            }
            trigger_tol *= 0.5; // f32 optimism: demand more before re-triggering
        }

        // restart rule (mirrors reference.rs)
        if e_cand.mu <= 0.5 * mu_last_restart || iters_since_restart >= 4096 {
            let mut enc = dev.create_command_encoder(&Default::default());
            if cand_is_avg {
                enc.copy_buffer_to_buffer(&xa, 0, x_cur, 0, (n * 4) as u64);
                enc.copy_buffer_to_buffer(&ya, 0, y_cur, 0, (m * 4) as u64);
            }
            enc.copy_buffer_to_buffer(x_cur, 0, &sum_x, 0, (n * 4) as u64);
            enc.copy_buffer_to_buffer(y_cur, 0, &sum_y, 0, (m * 4) as u64);
            ctx.queue.submit([enc.finish()]);
            count = 1;
            mu_last_restart = e_cand.mu;
            iters_since_restart = 0;
            restarts += 1;
        } else {
            iters_since_restart += opts.check_every as u64;
        }

        if let Some(limit) = opts.time_limit_ms {
            if start.elapsed().as_secs_f64() * 1000.0 > limit {
                status = SolveStatus::TimeLimit;
                break 'outer;
            }
        }
    }

    // non-optimal exit: read back current iterate, report honestly
    let (x_cur, y_cur): (&wgpu::Buffer, &wgpu::Buffer) = if parity == 0 {
        (&x, &y)
    } else {
        (&x_new, &y_new)
    };
    let xs = buffers::readback_f32(dev, &ctx.queue, x_cur, n).await;
    let ys = buffers::readback_f32(dev, &ctx.queue, y_cur, m).await;
    let xs64: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
    let ys64: Vec<f64> = ys.iter().map(|&v| v as f64).collect();
    let xo = s.unscale_x(&xs64);
    let mut yo = s.unscale_y(&ys64);
    project_dual(orig.row_lower, orig.row_upper, &mut yo);
    let verified = kkt::residuals_view(orig, &xo, &yo);
    Ok(finish(xo, yo, verified, status, iter, restarts, start))
}

/// Dispatch a prebuilt (pipeline, bind-group, workgroups) step into `enc`.
fn dispatch_prebuilt(
    enc: &mut wgpu::CommandEncoder,
    step: &(&wgpu::ComputePipeline, wgpu::BindGroup, u32),
) {
    let (pl, bg, w) = step;
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(pl);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(*w, 1, 1);
}

struct EvalBufs<'a> {
    c_s: &'a wgpu::Buffer,
    c_o: &'a wgpu::Buffer,
    lv_o: &'a wgpu::Buffer,
    uv_o: &'a wgpu::Buffer,
    lc_o: &'a wgpu::Buffer,
    uc_o: &'a wgpu::Buffer,
    dr: &'a wgpu::Buffer,
    dr_inv: &'a wgpu::Buffer,
    dc_inv: &'a wgpu::Buffer,
    ax_s: &'a wgpu::Buffer,
    ax_o: &'a wgpu::Buffer,
    aty_s: &'a wgpu::Buffer,
    aty_o: &'a wgpu::Buffer,
    rp: &'a wgpu::Buffer,
    rd: &'a wgpu::Buffer,
    bterm: &'a wgpu::Buffer,
    rterm: &'a wgpu::Buffer,
    y_o: &'a wgpu::Buffer,
}

/// Record the residual kernels + 5 reduction chains for candidate
/// (x_buf, y_buf) into `enc`. The two SpMVs are recorded by `gpu_op`; the
/// remaining six vector kernels stay engine-owned. Scalars land in
/// results[base..base+5]: [‖r_p‖², ‖r_d‖², Σ bterm, Σ rterm, c̄ᵀx̄].
#[allow(clippy::too_many_arguments)]
fn record_eval(
    dev: &wgpu::Device,
    enc: &mut wgpu::CommandEncoder,
    k: &Kernels,
    red: &Reducer,
    gpu_op: &dyn op::GpuOp,
    u_n: &wgpu::Buffer,
    u_m: &wgpu::Buffer,
    bufs: &EvalBufs<'_>,
    x_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
    n: usize,
    m: usize,
    results: &wgpu::Buffer,
    base: u32,
) {
    let disp =
        |enc: &mut wgpu::CommandEncoder, name: &str, entries: &[(u32, &wgpu::Buffer)], w: u32| {
            let pl = k.pipeline(name);
            let bg = kernels::bind(dev, pl, entries);
            kernels::pass_dispatch(enc, pl, &bg, w);
        };
    // primal residual chain: ax_s = A·x ; ax_o = ax_s ⊙ dr_inv ; rp = ax_o − clamp
    gpu_op.record_apply(dev, k, enc, x_buf, bufs.ax_s);
    disp(
        enc,
        "ew_mul",
        &[(0, u_m), (1, bufs.ax_s), (2, bufs.dr_inv), (6, bufs.ax_o)],
        wgs(m),
    );
    disp(
        enc,
        "primal_res",
        &[
            (0, u_m),
            (1, bufs.ax_o),
            (2, bufs.lc_o),
            (3, bufs.uc_o),
            (6, bufs.rp),
        ],
        wgs(m),
    );
    // dual residual chain: aty_s = Aᵀ·y ; aty_o = aty_s ⊙ dc_inv ; rd, bterm
    gpu_op.record_apply_t(dev, k, enc, y_buf, bufs.aty_s);
    disp(
        enc,
        "ew_mul",
        &[(0, u_n), (1, bufs.aty_s), (2, bufs.dc_inv), (6, bufs.aty_o)],
        wgs(n),
    );
    disp(
        enc,
        "dual_res_terms",
        &[
            (0, u_n),
            (1, bufs.aty_o),
            (2, bufs.c_o),
            (3, bufs.lv_o),
            (4, bufs.uv_o),
            (6, bufs.rd),
            (7, bufs.bterm),
        ],
        wgs(n),
    );
    // row (dual-objective) terms: y_o = y ⊙ dr ; rterm
    disp(
        enc,
        "ew_mul",
        &[(0, u_m), (1, y_buf), (2, bufs.dr), (6, bufs.y_o)],
        wgs(m),
    );
    disp(
        enc,
        "row_terms",
        &[
            (0, u_m),
            (1, bufs.y_o),
            (2, bufs.lc_o),
            (3, bufs.uc_o),
            (6, bufs.rterm),
        ],
        wgs(m),
    );

    red.record_dot(dev, enc, k, bufs.rp, bufs.rp, m, results, base);
    red.record_dot(dev, enc, k, bufs.rd, bufs.rd, n, results, base + 1);
    red.record_sum(dev, enc, k, bufs.bterm, n, results, base + 2);
    red.record_sum(dev, enc, k, bufs.rterm, m, results, base + 3);
    red.record_dot(dev, enc, k, bufs.c_s, x_buf, n, results, base + 4);
}

/// Convert packed scalar slots into relative residuals (host f64 math
/// identical to the old `eval` tail).
fn eval_from(vals: &[f32], base: usize, obj_offset: f64, q_norm: f64, c_norm: f64) -> EvalOut {
    let rp2 = vals[base] as f64;
    let rd2 = vals[base + 1] as f64;
    let bsum = vals[base + 2] as f64;
    let rsum = vals[base + 3] as f64;
    let cx = vals[base + 4] as f64;
    let rel_p = rp2.max(0.0).sqrt() / (1.0 + q_norm);
    let rel_d = rd2.max(0.0).sqrt() / (1.0 + c_norm);
    let pobj = cx + obj_offset;
    let dobj = obj_offset + bsum - rsum;
    let rel_gap = (pobj - dobj).abs() / (1.0 + pobj.abs() + dobj.abs());
    let mu = rel_p.max(rel_d).max(rel_gap);
    EvalOut {
        rel_p,
        rel_d,
        rel_gap,
        mu: if mu.is_finite() { mu } else { f64::INFINITY },
    }
}

/// Project the (f64) dual onto row-dual feasibility in place before the
/// authoritative `kkt::residuals` check. f32 iteration leaves ~1e-7 wrong-sign
/// noise on the duals of inactive rows; against an OPEN row bound that noise
/// drives `kkt::residuals`' dual objective to −∞ (NaN gap), which `mu()` then
/// silently drops — so the gap can never gate termination and the reported
/// objective stays as loose as bare primal feasibility. A feasible dual needs
/// `y_i ≤ 0` where the row has no upper bound and `y_i ≥ 0` where it has no
/// lower bound; snapping the noise to 0 restores the same finite, meaningful gap
/// the f64 reference (whose duals are already clean) converges on. The true dual
/// of an inactive constraint is exactly 0, so this only removes numerical dirt.
fn project_dual(row_lower: &[f64], row_upper: &[f64], y: &mut [f64]) {
    for (i, yi) in y.iter_mut().enumerate() {
        if !row_upper[i].is_finite() && *yi > 0.0 {
            *yi = 0.0;
        }
        if !row_lower[i].is_finite() && *yi < 0.0 {
            *yi = 0.0;
        }
    }
}

fn finish(
    x: Vec<f64>,
    y: Vec<f64>,
    verified: kkt::KktResiduals,
    status: SolveStatus,
    iterations: u64,
    restarts: u32,
    start: Instant,
) -> Solution {
    Solution {
        primal_obj: verified.primal_obj,
        x,
        y,
        status,
        stats: SolveStats {
            iterations,
            restarts,
            solve_ms: start.elapsed().as_secs_f64() * 1000.0,
            verified,
        },
    }
}
