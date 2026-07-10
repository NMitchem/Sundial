// Double-double (two-f32) accumulation variants of the accumulation-critical
// kernels (M2 df64 experiment). A df64 value is vec2<f32>(hi, lo) with
// |lo| ≤ ulp(hi)/2. Only ACCUMULATORS are df64; inputs/outputs stay f32.
// Error-free transforms (Knuth two_sum, fma-based two_prod) — these depend
// on IEEE-exact f32 ops; see the plan's fast-math risk note.
struct Params { n: u32, stride: u32, tau: f32, sigma: f32, w: f32, p0: f32, p1: f32, p2: f32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> in_a: array<f32>;
@group(0) @binding(2) var<storage, read> in_b: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_a: array<f32>;
@group(0) @binding(8) var<storage, read> idx_a: array<u32>;
@group(0) @binding(9) var<storage, read> idx_b: array<u32>;

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let bb = s - a;
    let err = (a - (s - bb)) + (b - bb);
    return vec2<f32>(s, err);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

fn df64_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    var s = two_sum(a.x, b.x);
    let lo = s.y + a.y + b.y;
    return two_sum(s.x, lo);
}

// acc += x·y in df64
fn df64_fma(acc: vec2<f32>, x: f32, y: f32) -> vec2<f32> {
    return df64_add(acc, two_prod(x, y));
}

// CSR SpMV with a df64 row accumulator. Same bindings as pdhg.wgsl spmv:
// idx_a=indptr idx_b=indices in_a=values in_b=x out_a=out (P.n = rows).
@compute @workgroup_size(256)
fn spmv_df64(@builtin(global_invocation_id) gid: vec3<u32>) {
    var r = gid.x;
    while (r < P.n) {
        var acc = vec2<f32>(0.0, 0.0);
        for (var k = idx_a[r]; k < idx_a[r + 1u]; k = k + 1u) {
            acc = df64_fma(acc, in_a[k], in_b[idx_b[k]]);
        }
        out_a[r] = acc.x + acc.y;
        r = r + P.stride;
    }
}

var<workgroup> sh2: array<vec2<f32>, 256>;

fn wg_tree_reduce_df64(lid: u32) {
    workgroupBarrier();
    var s = 128u;
    loop {
        if (lid < s) { sh2[lid] = df64_add(sh2[lid], sh2[lid + s]); }
        workgroupBarrier();
        s = s >> 1u;
        if (s == 0u) { break; }
    }
}

// partial dot(a,b) with df64 accumulation.
// in_a=a in_b=b out_a=partials[num_workgroups] (P.n = len, P.stride = threads)
@compute @workgroup_size(256)
fn reduce_dot_df64(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wid: vec3<u32>) {
    var acc = vec2<f32>(0.0, 0.0);
    var i = gid.x;
    while (i < P.n) {
        acc = df64_fma(acc, in_a[i], in_b[i]);
        i = i + P.stride;
    }
    sh2[lid.x] = acc;
    wg_tree_reduce_df64(lid.x);
    if (lid.x == 0u) { out_a[wid.x] = sh2[0].x + sh2[0].y; }
}

// partial sum(a) with df64 accumulation.
@compute @workgroup_size(256)
fn reduce_sum_df64(@builtin(global_invocation_id) gid: vec3<u32>,
                   @builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) wid: vec3<u32>) {
    var acc = vec2<f32>(0.0, 0.0);
    var i = gid.x;
    while (i < P.n) {
        acc = df64_add(acc, vec2<f32>(in_a[i], 0.0));
        i = i + P.stride;
    }
    sh2[lid.x] = acc;
    wg_tree_reduce_df64(lid.x);
    if (lid.x == 0u) { out_a[wid.x] = sh2[0].x + sh2[0].y; }
}

struct TParams { ns: u32, nt: u32, n: u32, stride: u32 }
@group(0) @binding(3) var<uniform> TP: TParams;
@group(0) @binding(4) var<storage, read> tsrc: array<f32>;
@group(0) @binding(7) var<storage, read_write> tdst: array<f32>;

// A·x for transport with df64 row accumulators (bindings 3/4/7 to avoid
// colliding with the shared Params block above; TransportGpuOp binds them).
@compute @workgroup_size(256)
fn ot_apply_df64(@builtin(global_invocation_id) gid: vec3<u32>) {
    var r = gid.x;
    let total = TP.ns + TP.nt;
    while (r < total) {
        var acc = vec2<f32>(0.0, 0.0);
        if (r < TP.ns) {
            let base = r * TP.nt;
            for (var j = 0u; j < TP.nt; j = j + 1u) {
                acc = df64_add(acc, vec2<f32>(tsrc[base + j], 0.0));
            }
        } else {
            let j = r - TP.ns;
            for (var i = 0u; i < TP.ns; i = i + 1u) {
                acc = df64_add(acc, vec2<f32>(tsrc[i * TP.nt + j], 0.0));
            }
        }
        tdst[r] = acc.x + acc.y;
        r = r + TP.stride;
    }
}
