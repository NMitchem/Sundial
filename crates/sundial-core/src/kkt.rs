use crate::problem::LpProblem;

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

pub fn residuals(p: &LpProblem, x: &[f64], y: &[f64]) -> KktResiduals {
    let (m, n) = (p.n_cons(), p.n_vars());
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), m);

    let mut ax = vec![0.0; m];
    p.a.mul(x, &mut ax);
    let mut aty = vec![0.0; n];
    p.at.mul(y, &mut aty);

    // primal residual
    let r_p = (0..m).map(|i| ax[i] - ax[i].clamp(p.row_lower[i], p.row_upper[i]));
    let q_norm = norm2((0..m).map(|i| {
        let l = if p.row_lower[i].is_finite() {
            p.row_lower[i].abs()
        } else {
            0.0
        };
        let u = if p.row_upper[i].is_finite() {
            p.row_upper[i].abs()
        } else {
            0.0
        };
        l.max(u)
    }));
    let rel_primal = norm2(r_p) / (1.0 + q_norm);

    // dual residual + dual objective bound-terms
    let g: Vec<f64> = (0..n).map(|j| p.c[j] + aty[j]).collect();
    let r_d = (0..n).map(|j| {
        let absorbed = (g[j] > 0.0 && p.col_lower[j].is_finite())
            || (g[j] < 0.0 && p.col_upper[j].is_finite());
        if absorbed {
            0.0
        } else {
            g[j]
        }
    });
    let rel_dual = norm2(r_d) / (1.0 + norm2(p.c.iter().copied()));

    let primal_obj = p.obj_offset + (0..n).map(|j| p.c[j] * x[j]).sum::<f64>();
    let bound_terms: f64 = (0..n)
        .map(|j| {
            if g[j] > 0.0 {
                g[j] * p.col_lower[j] // -inf here => dual obj -inf (honest)
            } else if g[j] < 0.0 {
                g[j] * p.col_upper[j]
            } else {
                0.0
            }
        })
        .sum();
    let row_terms: f64 = (0..m)
        .map(|i| {
            if y[i] > 0.0 {
                p.row_upper[i] * y[i]
            } else if y[i] < 0.0 {
                p.row_lower[i] * y[i]
            } else {
                0.0
            }
        })
        .sum();
    let dual_obj = p.obj_offset + bound_terms - row_terms;

    let rel_gap = (primal_obj - dual_obj).abs() / (1.0 + primal_obj.abs() + dual_obj.abs());
    KktResiduals {
        rel_primal,
        rel_dual,
        rel_gap,
        primal_obj,
        dual_obj,
    }
}
