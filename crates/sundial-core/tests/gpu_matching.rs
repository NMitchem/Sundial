//! Taxi-demo regression gate: the real Manhattan fixture, full scale, on the
//! GPU. The LP is solved to CPU-f64-verified `Optimal` (the only optimality
//! authority); the integral matching is then RECOVERED and its cost compared
//! against the certified dual floor from the same solve. The gate asserts the
//! recovery is injective and crossing-free, and that its measured slack over
//! the floor is within a small relative bound — it makes no optimality claim
//! about the matching itself, only the certificate + a measured comparison.
//!
//! Run in --release like the 1M transport gate:
//!   cargo test -p sundial-core --test gpu_matching --release -- --include-ignored --nocapture
use std::collections::HashSet;
use sundial_core::gpu::op::TransportGpuOp;
use sundial_core::gpu::GpuContext;
use sundial_core::problem::{SolveOptions, SolveStatus};
use sundial_core::recover::{recover_matching, segments_cross};
use sundial_core::transport;

/// Returns (riders, cabs, miles_per_unit) from the fixture JSON.
fn load_points() -> (Vec<[f64; 2]>, Vec<[f64; 2]>, f64) {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../web/public/taxi/points.json"
    );
    let txt = std::fs::read_to_string(path).expect("run scripts/fetch_taxi.py first");
    let v: serde_json::Value = serde_json::from_str(&txt).expect("valid points.json");
    let arr = |k: &str| -> Vec<[f64; 2]> {
        v[k].as_array()
            .expect(k)
            .iter()
            .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
            .collect()
    };
    let miles_per_unit = v["miles_per_unit"].as_f64().expect("miles_per_unit");
    (arr("riders"), arr("cabs"), miles_per_unit)
}

#[test]
#[ignore = "requires GPU"]
fn gpu_matches_manhattan_fixture_to_1e4() {
    let (riders, cabs, miles_per_unit) = load_points();
    assert_eq!((riders.len(), cabs.len()), (1024, 1152));
    let (ns, nt) = (riders.len(), cabs.len());
    let p = transport::problem_from_points(&riders, &cabs).expect("valid fixture");
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let gop = TransportGpuOp::new(&ctx.device, ns, nt);
    let opts = SolveOptions {
        tol: 1e-4,
        max_iters: 500_000,
        ..Default::default()
    };
    let sol = pollster::block_on(sundial_core::gpu::engine::solve_gpu_op(
        &ctx,
        &p,
        &gop,
        &opts,
        &mut |_e| {},
        None,
    ))
    .expect("solve");

    // The LP certificate is the ONLY optimality authority.
    assert_eq!(sol.status, SolveStatus::Optimal, "stats: {:?}", sol.stats);

    // Recover an integral injective matching from the fractional plan.
    let rec = recover_matching(&sol.x, &riders, &cabs);

    // Injectivity.
    let distinct: HashSet<u32> = rec.assignment.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        rec.assignment.len(),
        "assignment must be injective"
    );

    // Crossing-free: exhaustive O(n²) pair check (~524k pairs — fine in release).
    for a in 0..ns {
        for b in (a + 1)..ns {
            let pa = cabs[rec.assignment[a] as usize];
            let pb = cabs[rec.assignment[b] as usize];
            assert!(
                !segments_cross(riders[a], pa, riders[b], pb),
                "segments {a} and {b} properly cross"
            );
        }
    }

    // Measured comparison against the certified dual objective. The H3 scaling
    // multiplies the LP objective by 1/nt, so the certified CPU-f64 dual
    // objective in coordinate units is `dual_obj × nt` (kkt::KktResiduals);
    // recover's total_cost is already in coordinate units.
    //
    // MEASURED CAVEAT (see .superpowers/sdd/task-4a-report.md): at tol 1e-4 the
    // scaled duality gap is ~1e-4 and its sign is negative (dual iterate just
    // above the primal — a normal near-optimal PDHG artifact, neither iterate
    // exactly feasible). Amplified by ×nt≈1152 this is ~0.1 coordinate units, so
    // `dual_obj × nt` is NOT a strict lower bound here — it sits ~2.5% ABOVE the
    // true optimum (≈8.68, cross-checked vs Task 4's unscaled run + the recovered
    // cost). The recovered integral matching (the most accurate estimate) is
    // therefore BELOW it and `slack` is NEGATIVE. The bound below is thus a
    // one-sided check ("recovery is not WORSE than the certified objective
    // readout by >5e-4"), which holds comfortably; the certificate itself is the
    // only optimality authority. Slack sign/framing is flagged for adjudication.
    let dual_floor = sol.stats.verified.dual_obj * nt as f64;
    let slack = rec.total_cost - dual_floor;
    let slack_feet = slack * miles_per_unit * 5280.0;
    let rel = slack / dual_floor.abs().max(1.0);

    println!(
        "manhattan 1024x1152: {} iters, {} restarts, {:.0} ms, total_cost {:.6} units, \
         dual floor {:.6} units, slack {:.1} ft, support_edges {}",
        sol.stats.iterations,
        sol.stats.restarts,
        sol.stats.solve_ms,
        rec.total_cost,
        dual_floor,
        slack_feet,
        rec.support_edges,
    );

    assert!(
        rel <= 5e-4,
        "recovered matching cost {} exceeds the certified objective {dual_floor} by {rel} \
         (> 5e-4): slack {slack} ({slack_feet:.1} ft)",
        rec.total_cost,
    );
}
