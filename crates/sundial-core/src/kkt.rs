use crate::problem::{LpProblem, LpView};

#[derive(Debug, Clone, Copy)]
pub struct KktResiduals {
    pub rel_primal: f64,
    pub rel_dual: f64,
    pub rel_gap: f64,
    pub primal_obj: f64,
    pub dual_obj: f64,
}

impl KktResiduals {
    pub fn mu(&self) -> f64 {
        self.rel_primal.max(self.rel_dual).max(self.rel_gap)
    }
}

fn norm2(v: impl Iterator<Item = f64>) -> f64 {
    v.map(|a| a * a).sum::<f64>().sqrt()
}

/// Relative-residual denominators `(q_norm, c_norm)` for the given view,
/// extracted verbatim from `residuals_view()` so the engines can reuse the
/// exact f64 values (certificate space) and seed the primal weight
/// (iterate space).
pub fn denominators_view(v: &LpView) -> (f64, f64) {
    let m = v.op.n_rows();
    let q_norm = norm2((0..m).map(|i| {
        let l = if v.row_lower[i].is_finite() {
            v.row_lower[i].abs()
        } else {
            0.0
        };
        let u = if v.row_upper[i].is_finite() {
            v.row_upper[i].abs()
        } else {
            0.0
        };
        l.max(u)
    }));
    let c_norm = norm2(v.c.iter().copied());
    (q_norm, c_norm)
}

pub fn residuals(p: &LpProblem, x: &[f64], y: &[f64]) -> KktResiduals {
    residuals_view(&p.view(), x, y)
}

pub fn residuals_view(v: &LpView, x: &[f64], y: &[f64]) -> KktResiduals {
    let (m, n) = (v.op.n_rows(), v.op.n_cols());
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), m);

    let mut ax = vec![0.0; m];
    v.op.apply(x, &mut ax);
    let mut aty = vec![0.0; n];
    v.op.apply_t(y, &mut aty);

    let (q_norm, c_norm) = denominators_view(v);

    // primal residual
    let r_p = (0..m).map(|i| ax[i] - ax[i].clamp(v.row_lower[i], v.row_upper[i]));
    let rel_primal = norm2(r_p) / (1.0 + q_norm);

    // dual residual + dual objective bound-terms
    let g: Vec<f64> = (0..n).map(|j| v.c[j] + aty[j]).collect();
    let r_d = (0..n).map(|j| {
        let absorbed = (g[j] > 0.0 && v.col_lower[j].is_finite())
            || (g[j] < 0.0 && v.col_upper[j].is_finite());
        if absorbed {
            0.0
        } else {
            g[j]
        }
    });
    let rel_dual = norm2(r_d) / (1.0 + c_norm);

    let primal_obj = v.obj_offset + (0..n).map(|j| v.c[j] * x[j]).sum::<f64>();
    // Projected-multiplier dual objective: a term whose bound is non-finite
    // contributes 0 (the dual-infeasible component is projected out of D and
    // reported in r_d instead). See "Math conventions" — an f64 ±inf here
    // would make rel_gap NaN, which f64::max silently drops from mu.
    let bound_terms: f64 = (0..n)
        .map(|j| {
            if g[j] > 0.0 && v.col_lower[j].is_finite() {
                g[j] * v.col_lower[j]
            } else if g[j] < 0.0 && v.col_upper[j].is_finite() {
                g[j] * v.col_upper[j]
            } else {
                0.0
            }
        })
        .sum();
    let row_terms: f64 = (0..m)
        .map(|i| {
            if y[i] > 0.0 && v.row_upper[i].is_finite() {
                v.row_upper[i] * y[i]
            } else if y[i] < 0.0 && v.row_lower[i].is_finite() {
                v.row_lower[i] * y[i]
            } else {
                0.0
            }
        })
        .sum();
    let dual_obj = v.obj_offset + bound_terms - row_terms;

    let rel_gap = (primal_obj - dual_obj).abs() / (1.0 + primal_obj.abs() + dual_obj.abs());
    KktResiduals {
        rel_primal,
        rel_dual,
        rel_gap,
        primal_obj,
        dual_obj,
    }
}
