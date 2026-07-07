use js_sys::Function;
use serde::Serialize;
use sundial_core::gpu::{engine, GpuContext};
use sundial_core::problem::{ProgressEvent, SolveOptions, SolveStatus};
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct WasmResult {
    status: String,
    objective: f64,
    iterations: u64,
    restarts: u32,
    solve_ms: f64,
    rel_primal: f64,
    rel_dual: f64,
    rel_gap: f64,
    adapter: String,
}

/// Solve an MPS model on the browser's GPU. onProgress receives
/// {iter, rel_primal, rel_dual, rel_gap, ms_per_iter} every check interval.
#[wasm_bindgen(js_name = solveMps)]
pub async fn solve_mps(
    mps_text: String,
    tol: f64,
    on_progress: Function,
) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let p = crate::parse_str(&mps_text).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let ctx = GpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let opts = SolveOptions {
        tol,
        max_iters: 2_000_000,
        ..Default::default()
    };
    let mut cb = |e: ProgressEvent| {
        let v = serde_wasm_bindgen::to_value(&e).unwrap_or(JsValue::NULL);
        let _ = on_progress.call1(&JsValue::NULL, &v);
    };
    let sol = engine::solve_gpu(&ctx, &p, &opts, &mut cb)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let out = WasmResult {
        status: match sol.status {
            SolveStatus::Optimal => "Optimal (CPU f64 verified)".into(),
            other => format!("{other:?}"),
        },
        objective: sol.primal_obj,
        iterations: sol.stats.iterations,
        restarts: sol.stats.restarts,
        solve_ms: sol.stats.solve_ms,
        rel_primal: sol.stats.verified.rel_primal,
        rel_dual: sol.stats.verified.rel_dual,
        rel_gap: sol.stats.verified.rel_gap,
        adapter: ctx.adapter_name.clone(),
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}
