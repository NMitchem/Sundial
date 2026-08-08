//! A linear program small enough to check by hand.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example tiny_lp
//! ```
//!
//! A carpenter makes tables and chairs. Each table earns $5 and each chair
//! $3. A table needs 2 planks of wood and 1 hour of labor; a chair needs 1
//! plank and 2 hours. There are 10 planks and 8 hours available. How many of
//! each should they make?
//!
//! That question is a linear program: maximize profit, subject to not using
//! more wood or time than you have. Written out:
//!
//! ```text
//!   maximize    5·tables + 3·chairs
//!   subject to  2·tables + 1·chairs <= 10     (wood)
//!               1·tables + 2·chairs <=  8     (labor)
//!               tables, chairs >= 0
//! ```
//!
//! The answer is 4 tables and 2 chairs, for $26 — which uses every plank and
//! every hour exactly. Real problems have millions of variables instead of
//! two, which is the point of solving them on a GPU.

use sundial_core::problem::{CsrMatrix, LpProblem, SolveOptions, SolveStatus};

fn main() {
    // Sundial MINIMIZES, so maximizing 5t + 3c means minimizing -5t - 3c.
    let objective = vec![-5.0, -3.0];

    // The constraint coefficients, as a sparse (CSR) matrix:
    //
    //     [ 2  1 ]   <- wood:  2·tables + 1·chairs
    //     [ 1  2 ]   <- labor: 1·tables + 2·chairs
    //
    // CSR stores only non-zeros: `indptr` marks where each row starts in
    // `indices`/`values`. Every entry here happens to be non-zero, which is
    // unusual — real constraint matrices are mostly zeros, and that sparsity
    // is what makes large problems tractable.
    let constraints = CsrMatrix {
        n_rows: 2,
        n_cols: 2,
        indptr: vec![0, 2, 4],
        indices: vec![0, 1, 0, 1],
        values: vec![2.0, 1.0, 1.0, 2.0],
    };

    let problem = LpProblem::new(
        "carpenter".to_string(),
        constraints,
        objective,
        0.0,                        // constant added to the objective
        vec![f64::NEG_INFINITY; 2], // rows have no lower limit...
        vec![10.0, 8.0],            // ...and these upper limits
        vec![0.0, 0.0],             // can't make negative furniture
        vec![f64::INFINITY; 2],     // no upper limit on quantity
    )
    .expect("dimensions and bounds are consistent");

    // Solve on the CPU in f64. This path needs no GPU, so it runs anywhere.
    // The last argument is a progress callback; ignore it for a problem this
    // small. For the GPU, use `sundial_core::gpu::engine::solve_gpu`, which is
    // async — see `crates/sundial-cli/src/main.rs` for a worked call.
    let options = SolveOptions::default();
    let solution = sundial_core::reference::solve(&problem, &options, &mut |_| {});

    println!("status:  {:?}", solution.status);
    println!("tables:  {:.0}", solution.x[0]);
    println!("chairs:  {:.0}", solution.x[1]);
    println!("profit:  ${:.2}", -solution.primal_obj);
    println!(
        "\nsolved in {} iterations ({:.1} ms), verified on the CPU in f64 to {:.1e}",
        solution.stats.iterations,
        solution.stats.solve_ms,
        solution.stats.verified.mu(),
    );

    // `Optimal` is never returned on the solver's say-so alone: it means an
    // independent f64 KKT check passed at exactly the point printed above.
    assert_eq!(solution.status, SolveStatus::Optimal);
}
