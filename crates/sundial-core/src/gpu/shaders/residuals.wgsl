struct Params { n: u32, stride: u32, tau: f32, sigma: f32, w: f32, p0: f32, p1: f32, p2: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> in_a: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(3) var<storage, read> in_c: array<f32>;
@group(0) @binding(4) var<storage, read> in_d: array<f32>;
@group(0) @binding(5) var<storage, read> in_e: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_b: array<f32>;
@group(0) @binding(8) var<storage, read> idx_a: array<u32>;
@group(0) @binding(9) var<storage, read> idx_b: array<u32>;

const INF_THRESH: f32 = 0.5e30;

// rp = ax − clamp(ax, lc, uc)   in_a=ax_orig  in_b=lc  in_c=uc  out_a=rp
@compute @workgroup_size(256)
fn primal_res(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    let v = in_a[i];
    out_a[i] = v - clamp(v, in_b[i], in_c[i]);
}

// g = c + aty; rd per absorption rule; bterm for dual objective.
// PROJECTED-GAP SEMANTICS: when the bound a term would multiply is not finite,
// the contribution is 0 — this evaluates the dual objective at the sign-cone
// PROJECTION of the dual candidate (noise g>0 against l=-inf is dual-infeasible
// noise; rd still reports it, but the gap is computed as if it were projected
// to 0). Matches the CPU f64 gate, which verifies at the projected dual, and
// avoids all inf arithmetic in shaders (sentinel comparisons only).
// in_a=aty_orig  in_b=c  in_c=lv  in_d=uv  out_a=rd  out_b=bterm
@compute @workgroup_size(256)
fn dual_res_terms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= P.n) { return; }
    let g = in_b[j] + in_a[j];
    let l_fin = in_c[j] > -INF_THRESH;
    let u_fin = in_d[j] < INF_THRESH;
    var rd = g;
    if (g > 0.0 && l_fin) { rd = 0.0; }
    if (g < 0.0 && u_fin) { rd = 0.0; }
    out_a[j] = rd;
    var bt = 0.0;
    if (g > 0.0 && l_fin) { bt = g * in_c[j]; }
    if (g < 0.0 && u_fin) { bt = g * in_d[j]; }
    out_b[j] = bt;
}

// rterm[i] = y>0 ? uc·y : y<0 ? lc·y : 0, with the same projected-gap rule:
// a non-finite bound contributes 0 (dual-infeasible noise projected out).
// in_a=y_orig  in_b=lc  in_c=uc  out_a=rterm
@compute @workgroup_size(256)
fn row_terms(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    let y = in_a[i];
    let l_fin = in_b[i] > -INF_THRESH;
    let u_fin = in_c[i] < INF_THRESH;
    var t = 0.0;
    if (y > 0.0 && u_fin) { t = in_c[i] * y; }
    if (y < 0.0 && l_fin) { t = in_b[i] * y; }
    out_a[i] = t;
}
