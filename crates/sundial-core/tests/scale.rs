use sundial_core::{kkt, scale, testgen};

#[test]
fn ruiz_equilibrates_row_and_col_norms() {
    let (p, _, _, _) = testgen::generate(11, 50, 30);
    let (sp, _s) = scale::ruiz_pc(&p, 10);
    // Band rationale: Ruiz drives row/col inf-norms to ~1; Pock-Chambolle then
    // divides each entry by sqrt(row_sum)*sqrt(col_sum), both >= 1 post-Ruiz, so
    // entries can only SHRINK (upper bound is the real invariant: <= 1 + eps).
    // Lower bound: worst-case shrink is ~1/sqrt(nnz_row * nnz_col); 0.05 is a
    // safe floor for these test sizes (n=50, m=30, ~15% density).
    for r in 0..sp.a.n_rows {
        let mx = (sp.a.indptr[r] as usize..sp.a.indptr[r + 1] as usize)
            .map(|k| sp.a.values[k].abs())
            .fold(0.0f64, f64::max);
        assert!(
            mx > 0.05 && mx < 1.0 + 1e-9,
            "row {r} inf-norm {mx} not equilibrated"
        );
    }
    for ccol in 0..sp.at.n_rows {
        let mx = (sp.at.indptr[ccol] as usize..sp.at.indptr[ccol + 1] as usize)
            .map(|k| sp.at.values[k].abs())
            .fold(0.0f64, f64::max);
        assert!(
            mx > 0.05 && mx < 1.0 + 1e-9,
            "col {ccol} inf-norm {mx} not equilibrated"
        );
    }
}

#[test]
fn scaled_optimum_unscales_to_original_optimum() {
    // scale the problem, map the known optimum INTO scaled space, then back out;
    // KKT on the original problem must still be ~0 after the round trip.
    let (p, x, y, _) = testgen::generate(12, 40, 25);
    let (_sp, s) = scale::ruiz_pc(&p, 10);
    let x_scaled: Vec<f64> = (0..x.len()).map(|j| x[j] / s.col[j]).collect();
    let y_scaled: Vec<f64> = (0..y.len()).map(|i| y[i] / s.row[i]).collect();
    let x_back = s.unscale_x(&x_scaled);
    let y_back = s.unscale_y(&y_scaled);
    let r = kkt::residuals(&p, &x_back, &y_back);
    assert!(r.mu() < 1e-10, "round-trip mu = {}", r.mu());
}
