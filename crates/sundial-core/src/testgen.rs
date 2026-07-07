//! Constructed-KKT LP generator: builds (problem, x*, y*) satisfying the KKT
//! conditions exactly, giving an exact optimal oracle with no external solver.
//! Construction: sample sparse A and x*; pick per-column bound status
//! (interior / at-lower / at-upper) and per-row activity (inactive / active
//! upper y*>0 / active lower y*<0); set the active bound equal to (Ax*)ᵢ;
//! choose reduced costs g with signs matching column status; set c = g − Aᵀy*.
use crate::problem::{CsrMatrix, LpProblem};

pub fn generate(seed: u64, n: usize, m: usize) -> (LpProblem, Vec<f64>, Vec<f64>, f64) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let inf = f64::INFINITY;

    // sparse A: ~15% density, ensure every row and column nonempty
    let mut indptr = vec![0u32];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<f64> = Vec::new();
    for _r in 0..m {
        let mut cols: Vec<u32> = (0..n as u32).filter(|_| rng.f64() < 0.15).collect();
        if cols.is_empty() {
            cols.push(rng.u32(0..n as u32));
        }
        for &j in &cols {
            indices.push(j);
            values.push(rng.f64() * 4.0 - 2.0);
        }
        indptr.push(indices.len() as u32);
    }
    // ensure every column appears at least once
    let mut seen = vec![false; n];
    for &j in &indices {
        seen[j as usize] = true;
    }
    // (append a final row with the missing columns if any)
    let missing: Vec<u32> = (0..n as u32).filter(|&j| !seen[j as usize]).collect();
    let m = if missing.is_empty() {
        m
    } else {
        for &j in &missing {
            indices.push(j);
            values.push(1.0);
        }
        indptr.push(indices.len() as u32);
        m + 1
    };
    let a = CsrMatrix {
        n_rows: m,
        n_cols: n,
        indptr,
        indices,
        values,
    };

    let x: Vec<f64> = (0..n).map(|_| rng.f64() * 4.0 - 2.0).collect();
    let mut ax = vec![0.0; m];
    a.mul(&x, &mut ax);

    // column bounds + reduced-cost signs
    let mut col_lower = vec![0.0; n];
    let mut col_upper = vec![0.0; n];
    let mut g = vec![0.0; n];
    for j in 0..n {
        let u = rng.f64();
        if u < 0.4 {
            col_lower[j] = x[j] - 1.0 - rng.f64();
            col_upper[j] = x[j] + 1.0 + rng.f64();
            g[j] = 0.0; // interior
        } else if u < 0.7 {
            col_lower[j] = x[j]; // at lower bound
            col_upper[j] = if rng.bool() {
                x[j] + 1.0 + rng.f64()
            } else {
                inf
            };
            g[j] = 0.1 + 0.9 * rng.f64(); // g > 0 allowed at lower
        } else {
            col_upper[j] = x[j]; // at upper bound
            col_lower[j] = if rng.bool() {
                x[j] - 1.0 - rng.f64()
            } else {
                -inf
            };
            g[j] = -(0.1 + 0.9 * rng.f64()); // g < 0 allowed at upper
        }
    }

    // row bounds + duals
    let mut row_lower = vec![0.0; m];
    let mut row_upper = vec![0.0; m];
    let mut y = vec![0.0; m];
    for i in 0..m {
        let u = rng.f64();
        if u < 0.5 {
            // inactive: strict slack both sides (one side may be infinite)
            row_lower[i] = if rng.bool() {
                ax[i] - 1.0 - rng.f64()
            } else {
                -inf
            };
            row_upper[i] = if rng.bool() {
                ax[i] + 1.0 + rng.f64()
            } else {
                inf
            };
            if row_lower[i] == -inf && row_upper[i] == inf {
                row_upper[i] = ax[i] + 1.5;
            }
            y[i] = 0.0;
        } else if u < 0.75 {
            row_upper[i] = ax[i]; // active upper => y > 0
            row_lower[i] = if rng.bool() {
                ax[i] - 1.0 - rng.f64()
            } else {
                -inf
            };
            y[i] = 0.1 + 0.9 * rng.f64();
        } else {
            row_lower[i] = ax[i]; // active lower => y < 0
            row_upper[i] = if rng.bool() {
                ax[i] + 1.0 + rng.f64()
            } else {
                inf
            };
            y[i] = -(0.1 + 0.9 * rng.f64());
        }
    }

    // c = g − Aᵀ y*
    let at = a.transpose();
    let mut aty = vec![0.0; n];
    at.mul(&y, &mut aty);
    let c: Vec<f64> = (0..n).map(|j| g[j] - aty[j]).collect();

    let obj = (0..n).map(|j| c[j] * x[j]).sum::<f64>();
    let p = LpProblem::new(
        format!("testgen-{seed}"),
        a,
        c,
        0.0,
        row_lower,
        row_upper,
        col_lower,
        col_upper,
    )
    .expect("generator produced invalid problem");
    (p, x, y, obj)
}
