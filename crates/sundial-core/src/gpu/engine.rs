use super::buffers::{self, pack_f32_inf_sentinel};
use super::kernels::{self, Kernels, ParamsData, Reducer};
use super::{GpuContext, GpuError};
use crate::problem::*;
use crate::{kkt, reference, scale};
use web_time::Instant;

const WG: u32 = 256;
fn wgs(len: usize) -> u32 {
    (len as u32).div_ceil(WG).max(1)
}

struct EvalOut {
    rel_p: f64,
    rel_d: f64,
    rel_gap: f64,
    mu: f64,
}

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
    let start = Instant::now();
    let (sp, s) = scale::ruiz_pc(p, 10);
    let (m, n) = (sp.n_cons(), sp.n_vars());

    // buffer-size guard: largest single binding is the CSR values/indices array
    let needed_mib = ((sp.a.nnz() * 4).max((n.max(m)) * 4) as u64).div_ceil(1024 * 1024);
    if needed_mib > ctx.max_binding_mib {
        return Err(GpuError::BufferTooLarge {
            needed_mib,
            allowed_mib: ctx.max_binding_mib,
        });
    }

    let norm_a = reference::power_iteration_norm(&sp.a, &sp.at, 100, opts.seed);
    let (tau, sigma) = ((0.9 / norm_a) as f32, (0.9 / norm_a) as f32);
    let (q_norm, c_norm) = kkt::denominators(p); // ORIGINAL-space denominators, f64
    let dev = &ctx.device;
    let k = Kernels::new(dev);
    let red = Reducer::new(dev, n.max(m));

    // ---- pack + upload ----
    let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
    let b_f = |d: &[f32], l: &str| buffers::storage_f32(dev, d, l);
    let b_u = |d: &[u32], l: &str| buffers::storage_u32(dev, d, l);

    // scaled CSR A and Aᵀ
    let a_indptr = b_u(&sp.a.indptr, "a_indptr");
    let a_indices = b_u(&sp.a.indices, "a_indices");
    let a_vals = b_f(&f32v(&sp.a.values), "a_vals");
    let at_indptr = b_u(&sp.at.indptr, "at_indptr");
    let at_indices = b_u(&sp.at.indices, "at_indices");
    let at_vals = b_f(&f32v(&sp.at.values), "at_vals");

    // scaled iterate-space data (sentinel-packed bounds)
    let c_s = b_f(&f32v(&sp.c), "c_s");
    let lv_s = b_f(&pack_f32_inf_sentinel(&sp.col_lower), "lv_s");
    let uv_s = b_f(&pack_f32_inf_sentinel(&sp.col_upper), "uv_s");
    let lc_s = b_f(&pack_f32_inf_sentinel(&sp.row_lower), "lc_s");
    let uc_s = b_f(&pack_f32_inf_sentinel(&sp.row_upper), "uc_s");

    // ORIGINAL-space data for residual evaluation. Bounds use the SAME 1e30
    // sentinel as the iterate bounds (no true ±∞ in GPU buffers — WGSL may assume
    // finite math, so ±∞ is not portable across browser backends). The residual
    // kernels never multiply a non-finite bound: `dual_res_terms`/`row_terms` guard
    // every bound term by its INF_THRESH finiteness flag, computing the gap at the
    // sign-cone projection of the dual — identical semantics to the host
    // `project_dual` f64 gate, so the GPU trigger and the CPU verdict agree and
    // `rel_gap` stays finite for progress events.
    let c_o = b_f(&f32v(&p.c), "c_o");
    let lv_o = b_f(&pack_f32_inf_sentinel(&p.col_lower), "lv_o");
    let uv_o = b_f(&pack_f32_inf_sentinel(&p.col_upper), "uv_o");
    let lc_o = b_f(&pack_f32_inf_sentinel(&p.row_lower), "lc_o");
    let uc_o = b_f(&pack_f32_inf_sentinel(&p.row_upper), "uc_o");
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
    let x0: Vec<f32> = sp
        .col_lower
        .iter()
        .zip(&sp.col_upper)
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
    // parity 0 reads (x, y) writes (x_new, y_new); parity 1 the reverse.
    let make_iter =
        |x_src: &wgpu::Buffer, y_src: &wgpu::Buffer, x_dst: &wgpu::Buffer, y_dst: &wgpu::Buffer| {
            [
                // 1. aty = Aᵀ y_src            (spmv over n rows of Aᵀ)
                (
                    k.pipeline("spmv"),
                    kernels::bind(
                        dev,
                        k.pipeline("spmv"),
                        &[
                            (0, &u_n),
                            (8, &at_indptr),
                            (9, &at_indices),
                            (1, &at_vals),
                            (2, y_src),
                            (6, &aty),
                        ],
                    ),
                    wgs(n),
                ),
                // 2. x_dst, x_tilde = primal_step(x_src, aty)
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
                // 3. axt = A x_tilde
                (
                    k.pipeline("spmv"),
                    kernels::bind(
                        dev,
                        k.pipeline("spmv"),
                        &[
                            (0, &u_m),
                            (8, &a_indptr),
                            (9, &a_indices),
                            (1, &a_vals),
                            (2, &x_tilde),
                            (6, &axt),
                        ],
                    ),
                    wgs(m),
                ),
                // 4. y_dst = dual_step(y_src, axt)
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
                // 5. sum_x += x_dst ; 6. sum_y += y_dst
                (
                    k.pipeline("accum"),
                    kernels::bind(
                        dev,
                        k.pipeline("accum"),
                        &[(0, &u_n), (1, x_dst), (6, &sum_x)],
                    ),
                    wgs(n),
                ),
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

    let bufs = EvalBufs {
        a_indptr: &a_indptr,
        a_indices: &a_indices,
        a_vals: &a_vals,
        at_indptr: &at_indptr,
        at_indices: &at_indices,
        at_vals: &at_vals,
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
            for (pl, bg, w) in &iter_bg[parity] {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pl);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(*w, 1, 1);
            }
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

        // NaN/overflow watchdog
        let mx = red.maxabs(ctx, &k, x_cur, n).await;
        if !mx.is_finite() || mx > 1e29 {
            status = SolveStatus::NumericalBreakdown;
            break 'outer;
        }

        // averages: xa = sum_x / count, ya = sum_y / count
        {
            let u_avg_n = buffers::uniform_bytes(
                dev,
                &ParamsData {
                    n: n as u32,
                    stride: 0,
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
                    stride: 0,
                    tau,
                    sigma,
                    w: 1.0 / count as f32,
                }
                .bytes(),
                "u_avg_m",
            );
            let mut enc = dev.create_command_encoder(&Default::default());
            for (ub, src, dst, len) in [(&u_avg_n, &sum_x, &xa, n), (&u_avg_m, &sum_y, &ya, m)] {
                let pl = k.pipeline("ew_scale");
                let bg = kernels::bind(dev, pl, &[(0, ub), (1, src), (6, dst)]);
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wgs(len), 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
        }

        let e_cur = eval(
            ctx,
            &k,
            &red,
            &u_n,
            &u_m,
            &bufs,
            x_cur,
            y_cur,
            n,
            m,
            p.obj_offset,
            q_norm,
            c_norm,
        )
        .await;
        let e_avg = eval(
            ctx,
            &k,
            &red,
            &u_n,
            &u_m,
            &bufs,
            &xa,
            &ya,
            n,
            m,
            p.obj_offset,
            q_norm,
            c_norm,
        )
        .await;
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

        // termination trigger → authoritative CPU f64 verification
        if e_cand.mu <= trigger_tol {
            let xs = buffers::readback_f32(dev, &ctx.queue, cand_x, n).await;
            let ys = buffers::readback_f32(dev, &ctx.queue, cand_y, m).await;
            let xs64: Vec<f64> = xs.iter().map(|&v| v as f64).collect();
            let ys64: Vec<f64> = ys.iter().map(|&v| v as f64).collect();
            let xo = s.unscale_x(&xs64);
            let mut yo = s.unscale_y(&ys64);
            project_dual(p, &mut yo);
            let verified = kkt::residuals(p, &xo, &yo);
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
    project_dual(p, &mut yo);
    let verified = kkt::residuals(p, &xo, &yo);
    Ok(finish(xo, yo, verified, status, iter, restarts, start))
}

struct EvalBufs<'a> {
    a_indptr: &'a wgpu::Buffer,
    a_indices: &'a wgpu::Buffer,
    a_vals: &'a wgpu::Buffer,
    at_indptr: &'a wgpu::Buffer,
    at_indices: &'a wgpu::Buffer,
    at_vals: &'a wgpu::Buffer,
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

// ---- residual evaluation for a candidate (x_buf, y_buf) ----
// Returns f32 GPU metrics converted to f64 with host denominators.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
async fn eval(
    ctx: &GpuContext,
    k: &Kernels,
    red: &Reducer,
    u_n: &wgpu::Buffer,
    u_m: &wgpu::Buffer,
    bufs: &EvalBufs<'_>,
    x_buf: &wgpu::Buffer,
    y_buf: &wgpu::Buffer,
    n: usize,
    m: usize,
    obj_offset: f64,
    q_norm: f64,
    c_norm: f64,
) -> EvalOut {
    let dev = &ctx.device;
    let mut enc = dev.create_command_encoder(&Default::default());
    let steps: [(&str, Vec<(u32, &wgpu::Buffer)>, u32); 8] = [
        (
            "spmv",
            vec![
                (0, u_m),
                (8, bufs.a_indptr),
                (9, bufs.a_indices),
                (1, bufs.a_vals),
                (2, x_buf),
                (6, bufs.ax_s),
            ],
            wgs(m),
        ),
        (
            "ew_mul",
            vec![(0, u_m), (1, bufs.ax_s), (2, bufs.dr_inv), (6, bufs.ax_o)],
            wgs(m),
        ),
        (
            "primal_res",
            vec![
                (0, u_m),
                (1, bufs.ax_o),
                (2, bufs.lc_o),
                (3, bufs.uc_o),
                (6, bufs.rp),
            ],
            wgs(m),
        ),
        (
            "spmv",
            vec![
                (0, u_n),
                (8, bufs.at_indptr),
                (9, bufs.at_indices),
                (1, bufs.at_vals),
                (2, y_buf),
                (6, bufs.aty_s),
            ],
            wgs(n),
        ),
        (
            "ew_mul",
            vec![(0, u_n), (1, bufs.aty_s), (2, bufs.dc_inv), (6, bufs.aty_o)],
            wgs(n),
        ),
        (
            "dual_res_terms",
            vec![
                (0, u_n),
                (1, bufs.aty_o),
                (2, bufs.c_o),
                (3, bufs.lv_o),
                (4, bufs.uv_o),
                (6, bufs.rd),
                (7, bufs.bterm),
            ],
            wgs(n),
        ),
        (
            "ew_mul",
            vec![(0, u_m), (1, y_buf), (2, bufs.dr), (6, bufs.y_o)],
            wgs(m),
        ),
        (
            "row_terms",
            vec![
                (0, u_m),
                (1, bufs.y_o),
                (2, bufs.lc_o),
                (3, bufs.uc_o),
                (6, bufs.rterm),
            ],
            wgs(m),
        ),
    ];
    for (name, entries, w) in steps {
        let pl = k.pipeline(name);
        let bg = kernels::bind(dev, pl, &entries);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pl);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(w, 1, 1);
    }
    ctx.queue.submit([enc.finish()]);

    let rp2 = red.dot(ctx, k, bufs.rp, bufs.rp, m).await as f64;
    let rd2 = red.dot(ctx, k, bufs.rd, bufs.rd, n).await as f64;
    let bsum = red.sum(ctx, k, bufs.bterm, n).await as f64;
    let rsum = red.sum(ctx, k, bufs.rterm, m).await as f64;
    let cx = red.dot(ctx, k, bufs.c_s, x_buf, n).await as f64; // c̄ᵀx̄ = cᵀx

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
fn project_dual(p: &LpProblem, y: &mut [f64]) {
    for (i, yi) in y.iter_mut().enumerate() {
        if !p.row_upper[i].is_finite() && *yi > 0.0 {
            *yi = 0.0;
        }
        if !p.row_lower[i].is_finite() && *yi < 0.0 {
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
