//! Discrete optimal transport between two g×g grids as an LP — the M1 hero.
//! Constraint matrix is MATRIX-FREE (row/col sums via `TransportOp`); the
//! cost VECTOR is materialized (in-shader cost is M2 — adjudication in the
//! M1 plan). Cells: idx = row*g + col, center ((col+0.5)/g, (row+0.5)/g).
use crate::linop::LinOp;
use crate::problem::{CsrMatrix, LpProblem, OpProblem};

pub struct TransportOp {
    pub ns: usize,
    pub nt: usize,
}

impl LinOp for TransportOp {
    fn n_rows(&self) -> usize {
        self.ns + self.nt
    }
    fn n_cols(&self) -> usize {
        self.ns * self.nt
    }
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.ns {
            out[i] = x[i * self.nt..(i + 1) * self.nt].iter().sum();
        }
        for j in 0..self.nt {
            out[self.ns + j] = (0..self.ns).map(|i| x[i * self.nt + j]).sum();
        }
    }
    fn apply_t(&self, y: &[f64], out: &mut [f64]) {
        for i in 0..self.ns {
            let yi = y[i];
            for j in 0..self.nt {
                out[i * self.nt + j] = yi + y[self.ns + j];
            }
        }
    }
    /// A = [I⊗1ᵀ; 1ᵀ⊗I]: AᵀA = I⊗J + J⊗I, commuting, joint max eigenvalue
    /// ns + nt at the all-ones vector ⇒ ‖A‖₂ = √(ns+nt) exactly.
    fn norm2_exact(&self) -> Option<f64> {
        Some(((self.ns + self.nt) as f64).sqrt())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Preset {
    Blobs,
    Ring,
    Spiral,
    Checker,
    Corners,
}

impl std::str::FromStr for Preset {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "blobs" => Ok(Preset::Blobs),
            "ring" => Ok(Preset::Ring),
            "spiral" => Ok(Preset::Spiral),
            "checker" => Ok(Preset::Checker),
            "corners" => Ok(Preset::Corners),
            other => Err(format!(
                "unknown preset '{other}' (blobs|ring|spiral|checker|corners)"
            )),
        }
    }
}

fn gauss(px: f64, py: f64, cx: f64, cy: f64, s: f64) -> f64 {
    (-((px - cx).powi(2) + (py - cy).powi(2)) / (2.0 * s * s)).exp()
}

fn density(preset: Preset, source: bool, px: f64, py: f64) -> f64 {
    match (preset, source) {
        (Preset::Blobs, true) => gauss(px, py, 0.3, 0.3, 0.10) + gauss(px, py, 0.7, 0.7, 0.10),
        (Preset::Blobs, false) => gauss(px, py, 0.7, 0.3, 0.10) + gauss(px, py, 0.3, 0.7, 0.10),
        (Preset::Ring, true) => {
            let r = ((px - 0.5).powi(2) + (py - 0.5).powi(2)).sqrt();
            (-((r - 0.30) / 0.06).powi(2)).exp()
        }
        (Preset::Ring, false) => {
            if (0.3..=0.7).contains(&px) && (0.3..=0.7).contains(&py) {
                1.0
            } else {
                0.0
            }
        }
        (Preset::Spiral, true) => {
            // distance to a sampled Archimedean spiral around the center
            let mut d2min = f64::INFINITY;
            for t in 0..=255 {
                let s = t as f64 / 255.0;
                let theta = 4.0 * std::f64::consts::PI * s;
                let r = 0.08 + 0.30 * s;
                let (sx, sy) = (0.5 + r * theta.cos(), 0.5 + r * theta.sin());
                let d2 = (px - sx).powi(2) + (py - sy).powi(2);
                if d2 < d2min {
                    d2min = d2;
                }
            }
            (-d2min / (2.0 * 0.03 * 0.03)).exp()
        }
        (Preset::Spiral, false) => {
            // filled disk, radius 0.32
            let r = ((px - 0.5).powi(2) + (py - 0.5).powi(2)).sqrt();
            if r <= 0.32 {
                1.0
            } else {
                0.0
            }
        }
        (Preset::Checker, true) => {
            // 4×4 checkerboard, "black" squares carry the mass
            if ((px * 4.0).floor() as i64 + (py * 4.0).floor() as i64) % 2 == 0 {
                1.0
            } else {
                0.0
            }
        }
        (Preset::Checker, false) => {
            // the inverted board
            if ((px * 4.0).floor() as i64 + (py * 4.0).floor() as i64) % 2 == 0 {
                0.0
            } else {
                1.0
            }
        }
        (Preset::Corners, true) => {
            gauss(px, py, 0.15, 0.15, 0.07)
                + gauss(px, py, 0.85, 0.15, 0.07)
                + gauss(px, py, 0.15, 0.85, 0.07)
                + gauss(px, py, 0.85, 0.85, 0.07)
        }
        (Preset::Corners, false) => gauss(px, py, 0.5, 0.5, 0.12),
    }
}

/// Per-side masses on the g×g grid: density at cell centers, floored at
/// 1e-9 (no exactly-empty cells → no degenerate zero-mass equality rows),
/// normalized to sum exactly 1.0.
pub fn masses(preset: Preset, g: usize) -> (Vec<f64>, Vec<f64>) {
    let side = |source: bool| -> Vec<f64> {
        let mut v: Vec<f64> = (0..g * g)
            .map(|k| {
                let px = ((k % g) as f64 + 0.5) / g as f64;
                let py = ((k / g) as f64 + 0.5) / g as f64;
                density(preset, source, px, py).max(1e-9)
            })
            .collect();
        let sum: f64 = v.iter().sum();
        v.iter_mut().for_each(|x| *x /= sum);
        v
    };
    (side(true), side(false))
}

/// Squared Euclidean distance between cell centers, divided by g² so costs
/// stay in [0, 2] independent of grid size. Length g⁴ — g=32 ⇒ 8 MiB f64.
pub fn cost_vector(g: usize) -> Vec<f64> {
    let n = g * g;
    let mut c = vec![0.0; n * n];
    let coord = |k: usize| (((k % g) as f64 + 0.5), ((k / g) as f64 + 0.5));
    for i in 0..n {
        let (ix, iy) = coord(i);
        for j in 0..n {
            let (jx, jy) = coord(j);
            c[i * n + j] = ((ix - jx).powi(2) + (iy - jy).powi(2)) / (g * g) as f64;
        }
    }
    c
}

pub fn problem(preset: Preset, g: usize) -> OpProblem<TransportOp> {
    let (src, tgt) = masses(preset, g);
    let ns = g * g;
    let mut row = src;
    row.extend_from_slice(&tgt);
    OpProblem::new(
        format!("transport-{preset:?}-{g}x{g}").to_lowercase(),
        TransportOp { ns, nt: ns },
        cost_vector(g),
        0.0,
        row.clone(),
        row,
        vec![0.0; ns * ns],
        vec![f64::INFINITY; ns * ns],
    )
    .expect("transport problem is valid by construction")
}

/// Explicit CSR twin of `TransportOp` — TEST ORACLE ONLY (dense in memory
/// at O(ns·nt) nonzeros; never build at hero scale).
pub fn explicit_csr(ns: usize, nt: usize) -> CsrMatrix {
    let nnz = 2 * ns * nt;
    let mut indptr = Vec::with_capacity(ns + nt + 1);
    let mut indices = Vec::with_capacity(nnz);
    indptr.push(0u32);
    for i in 0..ns {
        for j in 0..nt {
            indices.push((i * nt + j) as u32);
        }
        indptr.push(indices.len() as u32);
    }
    for j in 0..nt {
        for i in 0..ns {
            indices.push((i * nt + j) as u32);
        }
        indptr.push(indices.len() as u32);
    }
    CsrMatrix {
        n_rows: ns + nt,
        n_cols: ns * nt,
        indptr,
        indices,
        values: vec![1.0; nnz],
    }
}

/// Explicit LpProblem twin of `problem()` — TEST ORACLE ONLY.
pub fn explicit_problem(preset: Preset, g: usize) -> LpProblem {
    let (src, tgt) = masses(preset, g);
    let ns = g * g;
    let mut row = src;
    row.extend_from_slice(&tgt);
    LpProblem::new(
        format!("transport-explicit-{g}x{g}"),
        explicit_csr(ns, ns),
        cost_vector(g),
        0.0,
        row.clone(),
        row,
        vec![0.0; ns * ns],
        vec![f64::INFINITY; ns * ns],
    )
    .expect("valid by construction")
}
