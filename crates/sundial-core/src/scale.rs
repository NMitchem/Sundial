//! Ruiz equilibration + Pock–Chambolle scaling, per the spec's conventions:
//! Ā = D_r·A·D_c, c̄ = D_c·c, row bounds ×D_r, col bounds ÷D_c.
//! Unscale: x = D_c·x̄, y = D_r·ȳ.
use crate::problem::{CsrMatrix, LpProblem};

#[derive(Debug, Clone)]
pub struct Scaling {
    pub row: Vec<f64>, // D_r diagonal
    pub col: Vec<f64>, // D_c diagonal
}

impl Scaling {
    /// No-op scaling for matrix-free problems (adjudication: they solve
    /// unscaled — the transport incidence structure is already balanced).
    pub fn identity(m: usize, n: usize) -> Self {
        Self {
            row: vec![1.0; m],
            col: vec![1.0; n],
        }
    }

    pub fn unscale_x(&self, x_scaled: &[f64]) -> Vec<f64> {
        x_scaled.iter().zip(&self.col).map(|(x, d)| x * d).collect()
    }
    pub fn unscale_y(&self, y_scaled: &[f64]) -> Vec<f64> {
        y_scaled.iter().zip(&self.row).map(|(y, d)| y * d).collect()
    }
}

pub fn ruiz_pc(p: &LpProblem, ruiz_iters: usize) -> (LpProblem, Scaling) {
    let (m, n) = (p.n_cons(), p.n_vars());
    let mut dr = vec![1.0f64; m];
    let mut dc = vec![1.0f64; n];
    let a = &p.a;

    // Ruiz: repeatedly divide each row/col by sqrt of its current inf-norm
    for _ in 0..ruiz_iters {
        let mut row_max = vec![0.0f64; m];
        let mut col_max = vec![0.0f64; n];
        for r in 0..m {
            for k in a.indptr[r] as usize..a.indptr[r + 1] as usize {
                let j = a.indices[k] as usize;
                let v = (a.values[k] * dr[r] * dc[j]).abs();
                if v > row_max[r] {
                    row_max[r] = v;
                }
                if v > col_max[j] {
                    col_max[j] = v;
                }
            }
        }
        for (d, &rm) in dr.iter_mut().zip(&row_max) {
            if rm > 0.0 {
                *d /= rm.sqrt();
            }
        }
        for (d, &cm) in dc.iter_mut().zip(&col_max) {
            if cm > 0.0 {
                *d /= cm.sqrt();
            }
        }
    }

    // Pock–Chambolle (alpha = 1): row scale 1/sqrt(sum_j |a_ij|), col 1/sqrt(sum_i |a_ij|)
    let mut row_sum = vec![0.0f64; m];
    let mut col_sum = vec![0.0f64; n];
    for r in 0..m {
        for k in a.indptr[r] as usize..a.indptr[r + 1] as usize {
            let j = a.indices[k] as usize;
            let v = (a.values[k] * dr[r] * dc[j]).abs();
            row_sum[r] += v;
            col_sum[j] += v;
        }
    }
    for (d, &rs) in dr.iter_mut().zip(&row_sum) {
        if rs > 0.0 {
            *d /= rs.sqrt();
        }
    }
    for (d, &cs) in dc.iter_mut().zip(&col_sum) {
        if cs > 0.0 {
            *d /= cs.sqrt();
        }
    }

    // build the scaled problem
    let mut values = p.a.values.clone();
    for (r, w) in p.a.indptr.windows(2).enumerate() {
        let (start, end) = (w[0] as usize, w[1] as usize);
        for (v, &j) in values[start..end].iter_mut().zip(&p.a.indices[start..end]) {
            *v *= dr[r] * dc[j as usize];
        }
    }
    let a_scaled = CsrMatrix {
        n_rows: m,
        n_cols: n,
        indptr: p.a.indptr.clone(),
        indices: p.a.indices.clone(),
        values,
    };
    let c: Vec<f64> = (0..n).map(|j| p.c[j] * dc[j]).collect();
    let row_lower: Vec<f64> = (0..m).map(|i| p.row_lower[i] * dr[i]).collect();
    let row_upper: Vec<f64> = (0..m).map(|i| p.row_upper[i] * dr[i]).collect();
    let col_lower: Vec<f64> = (0..n).map(|j| p.col_lower[j] / dc[j]).collect();
    let col_upper: Vec<f64> = (0..n).map(|j| p.col_upper[j] / dc[j]).collect();

    let scaled = LpProblem::new(
        p.name.clone(),
        a_scaled,
        c,
        p.obj_offset,
        row_lower,
        row_upper,
        col_lower,
        col_upper,
    )
    .expect("scaling preserved validity");
    (scaled, Scaling { row: dr, col: dc })
}
