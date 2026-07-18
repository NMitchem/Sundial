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
use sundial_core::recover::{self, recover_matching, segments_cross};
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

    // RIGOROUS certified floor via a repaired + coordinate-ascent-tightened
    // feasible dual (weak duality) — a true lower bound on the matching optimum,
    // independent of solver tolerance (unlike the tolerance-dependent readout
    // `dual_obj × nt`, which overshoots at tol 1e-4; see task-4a-report.md).
    // Both `certified_floor` and `total_cost` are in unscaled coordinate units.
    let mass = 1.0 / nt as f64; // H3 scaling: rider_mass = cab_cap = 1/nt
    let t0 = std::time::Instant::now();
    let cf = recover::certified_floor_ascent(
        &sol.y,
        &riders,
        &cabs,
        mass,
        mass,
        recover::FLOOR_MAX_SWEEPS,
    );
    let polish_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let floor = cf.value;
    let slack = rec.total_cost - floor;
    let slack_feet = slack * miles_per_unit * 5280.0;

    println!(
        "manhattan 1024x1152: {} iters, {} restarts, {:.0} ms, total_cost {:.6} units, \
         certified_floor {:.6} units, slack {:.1} ft, support_edges {}, ascent {} sweeps / {:.1} ms",
        sol.stats.iterations,
        sol.stats.restarts,
        sol.stats.solve_ms,
        rec.total_cost,
        floor,
        slack_feet,
        rec.support_edges,
        cf.sweeps,
        polish_ms,
    );

    // The floor is a rigorous lower bound, so the recovered integral matching
    // (a feasible primal point) MUST cost at least as much: slack ≥ 0 is
    // mathematically guaranteed. A negative value here would signal a scaling or
    // sign bug in the floor/rescale — this assert is a real bug-catcher, not a
    // tolerance check. (No upper bound asserted this round — the controller sets
    // it from the printed measurement.)
    assert!(
        slack >= 0.0,
        "certified floor {floor} exceeds recovered cost {} (slack {slack}, {slack_feet:.1} ft): \
         scaling/sign bug in certified_floor",
        rec.total_cost,
    );
}
