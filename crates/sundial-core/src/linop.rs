//! CPU-side linear-operator abstraction. PDHG and the KKT certificate only
//! ever need `A·x` and `Aᵀ·y`, so a problem may supply kernels instead of a
//! stored matrix (spec: "Matrix-free" problem form).
use crate::problem::CsrMatrix;

pub trait LinOp {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn apply(&self, x: &[f64], out: &mut [f64]);
    fn apply_t(&self, y: &[f64], out: &mut [f64]);
    /// Exact ‖A‖₂ when the structure makes it known (e.g. transport
    /// incidence: √(ns+nt)). None → estimate with power iteration.
    fn norm2_exact(&self) -> Option<f64> {
        None
    }
}

/// Borrowed explicit CSR pair as an operator.
pub struct CsrOp<'a> {
    pub a: &'a CsrMatrix,
    pub at: &'a CsrMatrix,
}

impl LinOp for CsrOp<'_> {
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

/// ‖A‖₂ via power iteration on AᵀA (Rayleigh quotient), operator form.
/// The explicit-matrix version delegates here, so results are bitwise
/// identical between the two forms for the same seed.
pub fn power_iteration_norm_op(op: &dyn LinOp, iters: usize, seed: u64) -> f64 {
    let mut rng = fastrand::Rng::with_seed(seed);
    let n = op.n_cols();
    let mut v: Vec<f64> = (0..n).map(|_| rng.f64() - 0.5).collect();
    let mut av = vec![0.0; op.n_rows()];
    let mut atav = vec![0.0; n];
    let mut lambda = 0.0f64;
    for _ in 0..iters {
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
        v.iter_mut().for_each(|x| *x /= norm);
        op.apply(&v, &mut av);
        op.apply_t(&av, &mut atav);
        lambda = v.iter().zip(&atav).map(|(a, b)| a * b).sum::<f64>();
        std::mem::swap(&mut v, &mut atav);
    }
    lambda.max(1e-30).sqrt()
}

pub fn op_norm2(op: &dyn LinOp, seed: u64) -> f64 {
    op.norm2_exact()
        .unwrap_or_else(|| power_iteration_norm_op(op, 100, seed))
}
