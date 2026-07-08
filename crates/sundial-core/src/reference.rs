//! CPU f64 reference implementation of restarted PDHG (the "Math conventions"
//! block of the M0 plan, executable). The GPU engine mirrors this loop exactly.
use crate::kkt::{self, KktResiduals};
use crate::problem::*;
use crate::scale;
use web_time::Instant;

pub fn power_iteration_norm(a: &CsrMatrix, at: &CsrMatrix, iters: usize, seed: u64) -> f64 {
    crate::linop::power_iteration_norm_op(&crate::linop::CsrOp { a, at }, iters, seed)
}

fn unscale(scaling: Option<&scale::Scaling>, x: &[f64], is_col: bool) -> Vec<f64> {
    match scaling {
        Some(s) if is_col => s.unscale_x(x),
        Some(s) => s.unscale_y(x),
        None => x.to_vec(),
    }
}

struct State {
    x: Vec<f64>,
    y: Vec<f64>,
    x_avg: Vec<f64>,
    y_avg: Vec<f64>,
    avg_count: u64,
}

impl State {
    /// Restart: optionally adopt the running average as the current iterate,
    /// then reset the average to the current iterate.
    fn restart_from(&mut self, from_avg: bool) {
        if from_avg {
            self.x.copy_from_slice(&self.x_avg);
            self.y.copy_from_slice(&self.y_avg);
        }
        self.x_avg.copy_from_slice(&self.x);
        self.y_avg.copy_from_slice(&self.y);
        self.avg_count = 1;
    }
}

pub fn solve(
    p: &LpProblem,
    opts: &SolveOptions,
    progress: &mut dyn FnMut(ProgressEvent),
) -> Solution {
    let (sp, s) = scale::ruiz_pc(p, 10);
    let norm_a = power_iteration_norm(&sp.a, &sp.at, 100, opts.seed);
    solve_view(&sp.view(), &p.view(), Some(&s), norm_a, opts, progress)
}

pub fn solve_op<O: crate::linop::LinOp>(
    p: &crate::problem::OpProblem<O>,
    opts: &SolveOptions,
    progress: &mut dyn FnMut(ProgressEvent),
) -> Solution {
    // Matrix-free problems solve UNSCALED (adjudication: the transport
    // operator is an all-ones incidence structure — already balanced).
    let norm_a = crate::linop::op_norm2(&p.op, opts.seed);
    let v = p.view();
    solve_view(&v, &v, None, norm_a, opts, progress)
}

pub fn solve_view(
    iterate: &LpView,
    original: &LpView,
    scaling: Option<&scale::Scaling>,
    norm_a: f64,
    opts: &SolveOptions,
    progress: &mut dyn FnMut(ProgressEvent),
) -> Solution {
    assert!(
        opts.check_every > 0,
        "SolveOptions::check_every must be > 0"
    );
    let start = Instant::now();
    let (m, n) = (iterate.op.n_rows(), iterate.op.n_cols());

    // Primal-weight balancing (ω) runs ONLY on the unscaled path (`scaling ==
    // None`, the matrix-free operator entry). The explicit path arrives
    // Ruiz+PC-equilibrated — that scaling already sets the primal/dual step
    // balance, so applying PDLP's ω = ‖c‖/‖q‖ on top double-corrects it. On the
    // unscaled transport problems ω is the only step balancing and is what
    // converges the 1M gate. Mirrors the GPU `solve_core` `primal_weight` gate.
    let mut omega = if scaling.is_none() {
        let (q_it, c_it) = kkt::denominators_view(iterate);
        crate::weight::initial_primal_weight(q_it, c_it)
    } else {
        1.0
    };
    let mut tau = 0.9 / (norm_a * omega);
    let mut sigma = 0.9 * omega / norm_a;

    let mut st = State {
        x: vec![0.0; n],
        y: vec![0.0; m],
        x_avg: vec![0.0; n],
        y_avg: vec![0.0; m],
        avg_count: 1,
    };
    // start feasible w.r.t. boxes: clamp 0 into [l_v, u_v]
    for j in 0..n {
        st.x[j] = 0.0f64.clamp(iterate.col_lower[j], iterate.col_upper[j]);
    }
    st.x_avg.copy_from_slice(&st.x);

    let mut aty = vec![0.0; n];
    let mut axt = vec![0.0; m];
    let mut x_new = vec![0.0; n];
    let mut x_tilde = vec![0.0; n];

    let mut mu_last_restart = f64::INFINITY;
    let mut iters_since_restart: u64 = 0;
    let mut restarts: u32 = 0;
    let mut status = SolveStatus::IterationLimit;
    let mut iter: u64 = 0;
    let mut last_check_time = Instant::now();
    let mut last_check_iter: u64 = 0;

    while iter < opts.max_iters {
        // one PDHG iteration (see Math conventions)
        iterate.op.apply_t(&st.y, &mut aty);
        for j in 0..n {
            let v = st.x[j] - tau * (iterate.c[j] + aty[j]);
            let xn = v.clamp(iterate.col_lower[j], iterate.col_upper[j]);
            x_new[j] = xn;
            x_tilde[j] = 2.0 * xn - st.x[j];
        }
        iterate.op.apply(&x_tilde, &mut axt);
        for (i, y_i) in st.y.iter_mut().enumerate() {
            let v = *y_i + sigma * axt[i];
            *y_i = v - sigma * (v / sigma).clamp(iterate.row_lower[i], iterate.row_upper[i]);
        }
        std::mem::swap(&mut st.x, &mut x_new);
        iter += 1;
        iters_since_restart += 1;

        // incremental running average
        st.avg_count += 1;
        let w = 1.0 / st.avg_count as f64;
        for j in 0..n {
            st.x_avg[j] += w * (st.x[j] - st.x_avg[j]);
        }
        for i in 0..m {
            st.y_avg[i] += w * (st.y[i] - st.y_avg[i]);
        }

        if iter.is_multiple_of(opts.check_every as u64) {
            if st.x.iter().any(|v| !v.is_finite()) || st.y.iter().any(|v| !v.is_finite()) {
                status = SolveStatus::NumericalBreakdown;
                break;
            }
            // IMPORTANT: residuals for termination/restart are evaluated on the
            // ORIGINAL problem (scaled-space residuals passing tol does NOT
            // imply the real ones do). Unscale candidates first.
            let r_cur = kkt::residuals_view(
                original,
                &unscale(scaling, &st.x, true),
                &unscale(scaling, &st.y, false),
            );
            let r_avg = kkt::residuals_view(
                original,
                &unscale(scaling, &st.x_avg, true),
                &unscale(scaling, &st.y_avg, false),
            );
            let (mu_cand, cand_is_avg) = if r_avg.mu() < r_cur.mu() {
                (r_avg.mu(), true)
            } else {
                (r_cur.mu(), false)
            };

            let now = Instant::now();
            let ms_per_iter = now.duration_since(last_check_time).as_secs_f64() * 1000.0
                / (iter - last_check_iter).max(1) as f64;
            last_check_time = now;
            last_check_iter = iter;
            progress(ProgressEvent {
                iter,
                rel_primal: r_cur.rel_primal,
                rel_dual: r_cur.rel_dual,
                rel_gap: r_cur.rel_gap,
                ms_per_iter,
            });

            if mu_cand <= opts.tol {
                if cand_is_avg {
                    st.restart_from(true);
                }
                status = SolveStatus::Optimal;
                break;
            }
            // restart rule
            if mu_cand <= 0.5 * mu_last_restart || iters_since_restart >= 4096 {
                st.restart_from(cand_is_avg);
                if scaling.is_none() {
                    let r_cand = if cand_is_avg { &r_avg } else { &r_cur };
                    omega = crate::weight::update_primal_weight(
                        omega,
                        r_cand.rel_primal,
                        r_cand.rel_dual,
                    );
                    tau = 0.9 / (norm_a * omega);
                    sigma = 0.9 * omega / norm_a;
                }
                mu_last_restart = mu_cand;
                iters_since_restart = 0;
                restarts += 1;
            }
            if let Some(limit) = opts.time_limit_ms {
                if start.elapsed().as_secs_f64() * 1000.0 > limit {
                    status = SolveStatus::TimeLimit;
                    break;
                }
            }
        }
    }

    // unscale and record the authoritative f64 verification on the ORIGINAL problem
    let x = unscale(scaling, &st.x, true);
    let y = unscale(scaling, &st.y, false);
    let verified: KktResiduals = kkt::residuals_view(original, &x, &y);
    assert!(
        status != SolveStatus::Optimal || verified.mu() <= opts.tol,
        "honesty violation: Optimal status with verified mu {} > tol {}",
        verified.mu(),
        opts.tol
    );
    let primal_obj = verified.primal_obj;
    Solution {
        x,
        y,
        primal_obj,
        status,
        stats: SolveStats {
            iterations: iter,
            restarts,
            solve_ms: start.elapsed().as_secs_f64() * 1000.0,
            verified,
        },
    }
}
