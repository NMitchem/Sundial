#![cfg(target_arch = "wasm32")]
//! sundial-lp: the wasm bindings for the Sundial GPU LP/optimal-transport
//! solver. Wasm-only by construction (empty crate on native targets) — see
//! sundial-core/sundial-mps for the portable engine and parser.

use js_sys::Function;
use serde::Serialize;
use sundial_core::gpu::op::TransportGpuOp;
use sundial_core::gpu::{engine, GpuContext};
use sundial_core::problem::{ProgressEvent, SnapshotEvent, Solution, SolveOptions, SolveStatus};
use sundial_core::recover;
use sundial_core::transport::{self, Preset};
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
    n_vars: u64,
}

fn make_result(sol: &Solution, adapter: &str, n_vars: usize) -> WasmResult {
    WasmResult {
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
        adapter: adapter.to_string(),
        n_vars: n_vars as u64,
    }
}

fn to_result(sol: &Solution, adapter: &str, n_vars: usize) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&make_result(sol, adapter, n_vars))
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

fn progress_cb(on_progress: Function) -> impl FnMut(ProgressEvent) {
    move |e: ProgressEvent| {
        let v = serde_wasm_bindgen::to_value(&e).unwrap_or(JsValue::NULL);
        let _ = on_progress.call1(&JsValue::NULL, &v);
    }
}

/// Capability probe: true when `navigator.gpu` exists. No device is
/// requested — callers can gate UI before committing to GPU init.
#[wasm_bindgen(js_name = webgpuAvailable)]
pub fn webgpu_available() -> bool {
    js_sys::Reflect::get(&js_sys::global(), &"navigator".into())
        .ok()
        .and_then(|nav| js_sys::Reflect::get(&nav, &"gpu".into()).ok())
        .map(|gpu| !gpu.is_undefined() && !gpu.is_null())
        .unwrap_or(false)
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
    let p = sundial_mps::parse_str(&mps_text).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let ctx = GpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let opts = SolveOptions {
        tol,
        max_iters: 2_000_000,
        ..Default::default()
    };
    let mut cb = progress_cb(on_progress);
    let n_vars = p.n_vars();
    let sol = engine::solve_gpu(&ctx, &p, &opts, &mut cb)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_result(&sol, &ctx.adapter_name, n_vars)
}

#[derive(Serialize)]
struct TransportPreview {
    grid: u32,
    src: Vec<f32>,
    tgt: Vec<f32>,
}

/// Masses for the hero heatmaps — no GPU, callable before solving.
#[wasm_bindgen(js_name = transportPreview)]
pub fn transport_preview(grid: u32, preset: String) -> Result<JsValue, JsValue> {
    let preset: Preset = preset.parse().map_err(|e: String| JsValue::from_str(&e))?;
    let (src, tgt) = transport::masses(preset, grid as usize);
    let out = TransportPreview {
        grid,
        src: src.iter().map(|&v| v as f32).collect(),
        tgt: tgt.iter().map(|&v| v as f32).collect(),
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Solve optimal transport on the browser's GPU. onSnapshot(iter, ax) gets
/// the achieved marginals (Float32Array, length 2·g²) every check interval.
#[wasm_bindgen(js_name = solveTransport)]
pub async fn solve_transport(
    grid: u32,
    preset: String,
    tol: f64,
    on_progress: Function,
    on_snapshot: Function,
) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let preset: Preset = preset.parse().map_err(|e: String| JsValue::from_str(&e))?;
    let g = grid as usize;
    let p = transport::problem(preset, g);
    let ctx = GpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let opts = SolveOptions {
        tol,
        max_iters: 500_000,
        ..Default::default()
    };
    let gop = TransportGpuOp::new(&ctx.device, g * g, g * g);
    let mut cb = progress_cb(on_progress);
    let mut snap = |s: SnapshotEvent| {
        let arr = js_sys::Float32Array::from(s.ax);
        let _ = on_snapshot.call2(&JsValue::NULL, &JsValue::from_f64(s.iter as f64), &arr);
    };
    let sol = engine::solve_gpu_op(&ctx, &p, &gop, &opts, &mut cb, Some(&mut snap))
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_result(&sol, &ctx.adapter_name, p.n_vars())
}

/// Solve optimal transport between hand-drawn masses (hero draw mode).
/// src/tgt are g² cell masses; junk values are cleaned and both sides
/// normalized (see transport::problem_from_masses).
#[wasm_bindgen(js_name = solveTransportCustom)]
pub async fn solve_transport_custom(
    grid: u32,
    src: Vec<f32>,
    tgt: Vec<f32>,
    tol: f64,
    on_progress: Function,
    on_snapshot: Function,
) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let g = grid as usize;
    let src64: Vec<f64> = src.iter().map(|&v| v as f64).collect();
    let tgt64: Vec<f64> = tgt.iter().map(|&v| v as f64).collect();
    let p = transport::problem_from_masses(&src64, &tgt64, g)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let ctx = GpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let opts = SolveOptions {
        tol,
        max_iters: 500_000,
        ..Default::default()
    };
    let gop = TransportGpuOp::new(&ctx.device, g * g, g * g);
    let mut cb = progress_cb(on_progress);
    let mut snap = |s: SnapshotEvent| {
        let arr = js_sys::Float32Array::from(s.ax);
        let _ = on_snapshot.call2(&JsValue::NULL, &JsValue::from_f64(s.iter as f64), &arr);
    };
    let sol = engine::solve_gpu_op(&ctx, &p, &gop, &opts, &mut cb, Some(&mut snap))
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_result(&sol, &ctx.adapter_name, p.n_vars())
}

#[derive(Serialize)]
struct MatchResult {
    // WasmResult fields listed FLAT — #[serde(flatten)] would route through
    // serialize_map, which serde_wasm_bindgen turns into an ES6 Map instead
    // of a plain object, silently breaking result.field access in JS
    // (Task 5 review finding). Flat fields ride the proven serialize_struct
    // path the four existing entry points use.
    status: String,
    objective: f64,
    iterations: u64,
    restarts: u32,
    solve_ms: f64,
    rel_primal: f64,
    rel_dual: f64,
    rel_gap: f64,
    adapter: String,
    n_vars: u64,
    /// Per-rider assigned cab (integral recovery from the LP plan's support:
    /// min-cost matching + uncrossing — injective and crossing-free).
    assignment: Vec<u32>,
    /// Total pickup distance of the recovered matching, coordinate units.
    total_cost: f64,
    /// Rigorous lower bound on ANY dispatch's total (repaired-dual weak
    /// duality + ascent) — display copy derives measured slack from this.
    certified_floor: f64,
    support_edges: u32,
}

/// Solve a rider→cab matching on the browser's GPU (taxi demo). riderXY and
/// cabXY are flat [x0,y0,x1,y1,…] in any consistent planar unit.
#[wasm_bindgen(js_name = solveMatching)]
pub async fn solve_matching(
    rider_xy: Vec<f32>,
    cab_xy: Vec<f32>,
    tol: f64,
    on_progress: Function,
) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    if !rider_xy.len().is_multiple_of(2) || !cab_xy.len().is_multiple_of(2) {
        return Err(JsValue::from_str("coordinate arrays must be [x,y] pairs"));
    }
    let to_pts = |v: &[f32]| -> Vec<[f64; 2]> {
        v.chunks_exact(2)
            .map(|p| [p[0] as f64, p[1] as f64])
            .collect()
    };
    let (riders, cabs) = (to_pts(&rider_xy), to_pts(&cab_xy));
    let p = transport::problem_from_points(&riders, &cabs)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let ctx = GpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let opts = SolveOptions {
        tol,
        max_iters: 500_000,
        ..Default::default()
    };
    let gop = TransportGpuOp::new(&ctx.device, riders.len(), cabs.len());
    let mut cb = progress_cb(on_progress);
    let n_vars = p.n_vars();
    let sol = engine::solve_gpu_op(&ctx, &p, &gop, &opts, &mut cb, None)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let rec = recover::recover_matching(&sol.x, &riders, &cabs);
    let mass = transport::matching_mass(cabs.len());
    let floor = recover::certified_floor(&sol.y, &riders, &cabs, mass, mass);
    let base = make_result(&sol, &ctx.adapter_name, n_vars);
    let out = MatchResult {
        status: base.status,
        objective: base.objective,
        iterations: base.iterations,
        restarts: base.restarts,
        solve_ms: base.solve_ms,
        rel_primal: base.rel_primal,
        rel_dual: base.rel_dual,
        rel_gap: base.rel_gap,
        adapter: base.adapter,
        n_vars: base.n_vars,
        assignment: rec.assignment,
        total_cost: rec.total_cost,
        certified_floor: floor,
        support_edges: rec.support_edges as u32,
    };
    serde_wasm_bindgen::to_value(&out).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Solve raw MPS bytes (plain or gzip) — the drop-a-file bench page path.
#[wasm_bindgen(js_name = solveMpsBytes)]
pub async fn solve_mps_bytes(
    bytes: Vec<u8>,
    tol: f64,
    on_progress: Function,
) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let p = sundial_mps::parse_bytes(&bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
    let ctx = GpuContext::new()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let opts = SolveOptions {
        tol,
        max_iters: 2_000_000,
        ..Default::default()
    };
    let mut cb = progress_cb(on_progress);
    let n_vars = p.n_vars();
    let sol = engine::solve_gpu(&ctx, &p, &opts, &mut cb)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    to_result(&sol, &ctx.adapter_name, n_vars)
}
