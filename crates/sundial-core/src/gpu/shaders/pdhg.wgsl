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

// x⁺ = clamp(x − τ(c + aᵀy)); x̃ = 2x⁺ − x
// in_a=x  in_b=aty  in_c=c  in_d=lv  in_e=uv  out_a=x_new  out_b=x_tilde
@compute @workgroup_size(256)
fn primal_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= P.n) { return; }
    let xj = in_a[j];
    let v = xj - P.tau * (in_c[j] + in_b[j]);
    let xn = clamp(v, in_d[j], in_e[j]);
    out_a[j] = xn;
    out_b[j] = 2.0 * xn - xj;
}

// v = y + σ(ax̃); y⁺ = v − σ·clamp(v/σ, lc, uc)
// in_a=y  in_b=axt  in_c=lc  in_d=uc  out_a=y_new
@compute @workgroup_size(256)
fn dual_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    let v = in_a[i] + P.sigma * in_b[i];
    out_a[i] = v - P.sigma * clamp(v / P.sigma, in_c[i], in_d[i]);
}

// CSR SpMV, one row per thread.
// idx_a=indptr  idx_b=indices  in_a=values  in_b=x  out_a=out   (P.n = rows)
@compute @workgroup_size(256)
fn spmv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= P.n) { return; }
    var acc = 0.0;
    for (var k = idx_a[r]; k < idx_a[r + 1u]; k = k + 1u) {
        acc = acc + in_a[k] * in_b[idx_b[k]];
    }
    out_a[r] = acc;
}

// out += in (running-sum accumulation for iterate averaging)   in_a=src  out_a=sum
@compute @workgroup_size(256)
fn accum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    out_a[i] = out_a[i] + in_a[i];
}

// out = in · w (divide running sum by count at check time)   in_a=sum  out_a=avg
@compute @workgroup_size(256)
fn ew_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    out_a[i] = in_a[i] * P.w;
}

// out = a ⊙ b (elementwise multiply; used for row/col unscaling)
// in_a=v  in_b=scale  out_a=out
@compute @workgroup_size(256)
fn ew_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= P.n) { return; }
    out_a[i] = in_a[i] * in_b[i];
}
