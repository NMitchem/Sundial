use std::collections::HashMap;
use std::path::PathBuf;
use sundial_core::problem::{SolveOptions, SolveStatus};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sundial-mps/tests/fixtures")
}

fn optima() -> HashMap<String, f64> {
    let csv = include_str!("../data/netlib_optima.csv");
    csv.lines()
        .skip(1)
        .map(|l| {
            let (name, v) = l.split_once(',').unwrap();
            (name.to_string(), v.parse().unwrap())
        })
        .collect()
}

fn check(name: &str, sol: &sundial_core::problem::Solution, want: f64, tol: f64) {
    assert_eq!(
        sol.status,
        SolveStatus::Optimal,
        "{name}: status {:?}",
        sol.status
    );
    assert!(
        sol.stats.verified.mu() <= tol,
        "{name}: verified mu {}",
        sol.stats.verified.mu()
    );
    let rel = (sol.primal_obj - want).abs() / (1.0 + want.abs());
    assert!(
        rel <= 1e-3,
        "{name}: obj {} vs known {want} (rel {rel:.2e})",
        sol.primal_obj
    );
}

#[test]
fn cpu_reference_solves_netlib_smalls_to_1e4() {
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 2_000_000,
        ..Default::default()
    };
    let optima = optima();
    for name in ["afiro", "sc50a", "sc50b"] {
        let bytes = std::fs::read(fixture_dir().join(format!("{name}.mps"))).unwrap();
        let p = sundial_mps::parse_bytes(&bytes).unwrap();
        let sol = sundial_core::reference::solve(&p, &opts, &mut |_| {});
        check(name, &sol, optima[name], opts.tol);
    }
}

#[test]
#[ignore = "requires GPU"]
fn gpu_solves_netlib_fixtures_to_1e4() {
    let ctx = pollster::block_on(sundial_core::gpu::GpuContext::new()).expect("no GPU");
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 2_000_000,
        ..Default::default()
    };
    let optima = optima();
    for name in ["afiro", "sc50a", "sc50b", "adlittle", "share2b"] {
        let bytes = std::fs::read(fixture_dir().join(format!("{name}.mps"))).unwrap();
        let p = sundial_mps::parse_bytes(&bytes).unwrap();
        let sol = pollster::block_on(sundial_core::gpu::engine::solve_gpu(
            &ctx,
            &p,
            &opts,
            &mut |_| {},
        ))
        .unwrap();
        check(name, &sol, optima[name], opts.tol);
    }
}
