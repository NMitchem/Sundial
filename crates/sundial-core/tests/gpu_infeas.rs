use sundial_core::gpu::{engine, GpuContext};
use sundial_core::problem::{CsrMatrix, LpProblem, SolveOptions, SolveStatus};

fn infeasible_lp() -> LpProblem {
    let a = CsrMatrix {
        n_rows: 2,
        n_cols: 1,
        indptr: vec![0, 1, 2],
        indices: vec![0, 0],
        values: vec![1.0, 1.0],
    };
    LpProblem::new(
        "infeas-2row".into(),
        a,
        vec![0.0],
        0.0,
        vec![1.0, f64::NEG_INFINITY],
        vec![f64::INFINITY, 0.0],
        vec![f64::NEG_INFINITY],
        vec![f64::INFINITY],
    )
    .unwrap()
}

fn unbounded_lp() -> LpProblem {
    let a = CsrMatrix {
        n_rows: 1,
        n_cols: 1,
        indptr: vec![0, 1],
        indices: vec![0],
        values: vec![1.0],
    };
    LpProblem::new(
        "unbounded-1var".into(),
        a,
        vec![-1.0],
        0.0,
        vec![0.0],
        vec![f64::INFINITY],
        vec![0.0],
        vec![f64::INFINITY],
    )
    .unwrap()
}

#[test]
#[ignore = "requires GPU"]
fn gpu_detects_infeasible() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = pollster::block_on(engine::solve_gpu(
        &ctx,
        &infeasible_lp(),
        &opts,
        &mut |_| {},
    ))
    .unwrap();
    assert_eq!(
        sol.status,
        SolveStatus::Infeasible,
        "iters {}",
        sol.stats.iterations
    );
}

#[test]
#[ignore = "requires GPU"]
fn gpu_detects_unbounded() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol =
        pollster::block_on(engine::solve_gpu(&ctx, &unbounded_lp(), &opts, &mut |_| {})).unwrap();
    assert_eq!(
        sol.status,
        SolveStatus::Unbounded,
        "iters {}",
        sol.stats.iterations
    );
}

/// Field data: netlib's infeasible set (fetched by scripts/fetch_netlib.sh
/// scripts/netlib_infeas.txt infeas). HARD assertion: never Optimal.
/// SOFT: report how many certify Infeasible (heuristic recall is not gated).
#[test]
#[ignore = "requires GPU"]
fn netlib_infeas_set_never_optimal() {
    let dir = std::path::Path::new("../../bench/infeas");
    if !dir.exists() {
        eprintln!("bench/infeas not fetched — run: bash scripts/fetch_netlib.sh scripts/netlib_infeas.txt infeas");
        return;
    }
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 300_000,
        ..Default::default()
    };
    let (mut certified, mut total) = (0, 0);
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "mps") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let bytes = std::fs::read(&path).unwrap();
        let p = match sundial_mps::parse_bytes(&bytes) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{name}: parse error ({e}) — skipped");
                continue;
            }
        };
        total += 1;
        let sol = pollster::block_on(engine::solve_gpu(&ctx, &p, &opts, &mut |_| {})).unwrap();
        eprintln!("{name}: {:?} ({} iters)", sol.status, sol.stats.iterations);
        assert_ne!(
            sol.status,
            SolveStatus::Optimal,
            "{name}: FALSE OPTIMAL on an infeasible instance"
        );
        if sol.status == SolveStatus::Infeasible {
            certified += 1;
        }
    }
    eprintln!("infeas set: {certified}/{total} certified Infeasible (rest honest IterationLimit)");
    assert!(total > 0, "no instances parsed");
}
