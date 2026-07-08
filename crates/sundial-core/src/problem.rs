use thiserror::Error;

#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub indptr: Vec<u32>,  // len n_rows + 1
    pub indices: Vec<u32>, // len nnz
    pub values: Vec<f64>,  // len nnz
}

impl CsrMatrix {
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn mul(&self, x: &[f64], out: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols);
        assert_eq!(out.len(), self.n_rows);
        for (r, out_r) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for k in self.indptr[r] as usize..self.indptr[r + 1] as usize {
                acc += self.values[k] * x[self.indices[k] as usize];
            }
            *out_r = acc;
        }
    }

    pub fn transpose(&self) -> CsrMatrix {
        // counting sort by column
        let mut counts = vec![0u32; self.n_cols + 1];
        for &j in &self.indices {
            counts[j as usize + 1] += 1;
        }
        for j in 0..self.n_cols {
            counts[j + 1] += counts[j];
        }
        let indptr = counts.clone();
        let mut pos = counts;
        let nnz = self.nnz();
        let mut indices = vec![0u32; nnz];
        let mut values = vec![0.0f64; nnz];
        for r in 0..self.n_rows {
            for k in self.indptr[r] as usize..self.indptr[r + 1] as usize {
                let j = self.indices[k] as usize;
                let dst = pos[j] as usize;
                indices[dst] = r as u32;
                values[dst] = self.values[k];
                pos[j] += 1;
            }
        }
        CsrMatrix {
            n_rows: self.n_cols,
            n_cols: self.n_rows,
            indptr,
            indices,
            values,
        }
    }
}

#[derive(Debug, Error)]
pub enum ProblemError {
    #[error("dimension mismatch: {0}")]
    Dimension(String),
    #[error("bound l > u at {kind} index {index}: {l} > {u}")]
    BoundOrder {
        kind: &'static str,
        index: usize,
        l: f64,
        u: f64,
    },
}

#[derive(Debug, Clone)]
pub struct LpProblem {
    pub name: String,
    pub a: CsrMatrix,
    pub at: CsrMatrix,
    pub c: Vec<f64>,
    pub obj_offset: f64,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
}

impl LpProblem {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        a: CsrMatrix,
        c: Vec<f64>,
        obj_offset: f64,
        row_lower: Vec<f64>,
        row_upper: Vec<f64>,
        col_lower: Vec<f64>,
        col_upper: Vec<f64>,
    ) -> Result<Self, ProblemError> {
        let (m, n) = (a.n_rows, a.n_cols);
        if c.len() != n || col_lower.len() != n || col_upper.len() != n {
            return Err(ProblemError::Dimension(format!(
                "n={n} but c/col bounds differ"
            )));
        }
        if row_lower.len() != m || row_upper.len() != m {
            return Err(ProblemError::Dimension(format!(
                "m={m} but row bounds differ"
            )));
        }
        for i in 0..m {
            if row_lower[i] > row_upper[i] {
                return Err(ProblemError::BoundOrder {
                    kind: "row",
                    index: i,
                    l: row_lower[i],
                    u: row_upper[i],
                });
            }
        }
        for j in 0..n {
            if col_lower[j] > col_upper[j] {
                return Err(ProblemError::BoundOrder {
                    kind: "col",
                    index: j,
                    l: col_lower[j],
                    u: col_upper[j],
                });
            }
        }
        let at = a.transpose();
        Ok(Self {
            name,
            a,
            at,
            c,
            obj_offset,
            row_lower,
            row_upper,
            col_lower,
            col_upper,
        })
    }

    pub fn n_vars(&self) -> usize {
        self.a.n_cols
    }
    pub fn n_cons(&self) -> usize {
        self.a.n_rows
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolveStatus {
    Optimal,
    IterationLimit,
    TimeLimit,
    NumericalBreakdown,
}

#[derive(Debug, Clone)]
pub struct SolveStats {
    pub iterations: u64,
    pub restarts: u32,
    pub solve_ms: f64,
    pub verified: crate::kkt::KktResiduals, // CPU f64, on the ORIGINAL problem
}

#[derive(Debug, Clone)]
pub struct Solution {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub primal_obj: f64,
    pub status: SolveStatus,
    pub stats: SolveStats,
}

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub tol: f64,
    pub max_iters: u64,
    pub time_limit_ms: Option<f64>,
    pub check_every: u32,
    pub seed: u64,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            tol: 1e-4,
            max_iters: 500_000,
            time_limit_ms: None,
            check_every: 64,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(target_arch = "wasm32", derive(serde::Serialize))]
pub struct ProgressEvent {
    pub iter: u64,
    pub rel_primal: f64,
    pub rel_dual: f64,
    pub rel_gap: f64,
    pub ms_per_iter: f64,
}

impl crate::linop::LinOp for LpProblem {
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

/// Borrowed, operator-generic view of a problem: everything the solver loop
/// and the KKT certificate need, with the matrix behind `LinOp`.
pub struct LpView<'a> {
    pub op: &'a dyn crate::linop::LinOp,
    pub c: &'a [f64],
    pub obj_offset: f64,
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
}

impl LpProblem {
    pub fn view(&self) -> LpView<'_> {
        LpView {
            op: self,
            c: &self.c,
            obj_offset: self.obj_offset,
            row_lower: &self.row_lower,
            row_upper: &self.row_upper,
            col_lower: &self.col_lower,
            col_upper: &self.col_upper,
        }
    }
}

/// A problem whose constraint matrix is an operator (never materialized).
pub struct OpProblem<O: crate::linop::LinOp> {
    pub name: String,
    pub op: O,
    pub c: Vec<f64>,
    pub obj_offset: f64,
    pub row_lower: Vec<f64>,
    pub row_upper: Vec<f64>,
    pub col_lower: Vec<f64>,
    pub col_upper: Vec<f64>,
}

impl<O: crate::linop::LinOp> OpProblem<O> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: String,
        op: O,
        c: Vec<f64>,
        obj_offset: f64,
        row_lower: Vec<f64>,
        row_upper: Vec<f64>,
        col_lower: Vec<f64>,
        col_upper: Vec<f64>,
    ) -> Result<Self, ProblemError> {
        let (m, n) = (op.n_rows(), op.n_cols());
        if c.len() != n || col_lower.len() != n || col_upper.len() != n {
            return Err(ProblemError::Dimension(format!(
                "n={n} but c/col bounds differ"
            )));
        }
        if row_lower.len() != m || row_upper.len() != m {
            return Err(ProblemError::Dimension(format!(
                "m={m} but row bounds differ"
            )));
        }
        for i in 0..m {
            if row_lower[i] > row_upper[i] {
                return Err(ProblemError::BoundOrder {
                    kind: "row",
                    index: i,
                    l: row_lower[i],
                    u: row_upper[i],
                });
            }
        }
        for j in 0..n {
            if col_lower[j] > col_upper[j] {
                return Err(ProblemError::BoundOrder {
                    kind: "col",
                    index: j,
                    l: col_lower[j],
                    u: col_upper[j],
                });
            }
        }
        Ok(Self {
            name,
            op,
            c,
            obj_offset,
            row_lower,
            row_upper,
            col_lower,
            col_upper,
        })
    }

    pub fn view(&self) -> LpView<'_> {
        LpView {
            op: &self.op,
            c: &self.c,
            obj_offset: self.obj_offset,
            row_lower: &self.row_lower,
            row_upper: &self.row_upper,
            col_lower: &self.col_lower,
            col_upper: &self.col_upper,
        }
    }

    pub fn n_vars(&self) -> usize {
        self.op.n_cols()
    }
    pub fn n_cons(&self) -> usize {
        self.op.n_rows()
    }
}
