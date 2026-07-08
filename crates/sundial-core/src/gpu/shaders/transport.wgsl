// Matrix-free optimal-transport incidence operator (M1 plan "Math
// conventions"). X ∈ R^(ns×nt) flattened row-major; A = [rows; cols].
// Own uniform struct — auto pipeline layout keeps this independent of the
// shared Params convention in pdhg.wgsl.
struct TParams { ns: u32, nt: u32, n: u32, stride: u32 }
@group(0) @binding(0) var<uniform> P: TParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(6) var<storage, read_write> dst: array<f32>;

// A·x: dst[i] = Σ_j x[i·nt+j] for i < ns; dst[ns+j] = Σ_i x[i·nt+j].
// One thread per output row (m = ns+nt rows), grid-stride.
// Plain f32 accumulation: ≤4096-term sums of same-sign masses carry a
// ~1e-6 relative error — two decades under the 1e-4 tier. Revisit with
// Neumaier compensation only if the verified-residual floor demands it.
@compute @workgroup_size(256)
fn ot_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    var r = gid.x;
    let total = P.ns + P.nt;
    while (r < total) {
        var acc = 0.0;
        if (r < P.ns) {
            let base = r * P.nt;
            for (var j = 0u; j < P.nt; j = j + 1u) {
                acc = acc + src[base + j];
            }
        } else {
            let j = r - P.ns;
            for (var i = 0u; i < P.ns; i = i + 1u) {
                acc = acc + src[i * P.nt + j];
            }
        }
        dst[r] = acc;
        r = r + P.stride;
    }
}

// Aᵀ·y: dst[i·nt+j] = y[i] + y[ns+j]. Grid-stride over n = ns·nt.
@compute @workgroup_size(256)
fn ot_apply_t(@builtin(global_invocation_id) gid: vec3<u32>) {
    var k = gid.x;
    while (k < P.n) {
        dst[k] = src[k / P.nt] + src[P.ns + k % P.nt];
        k = k + P.stride;
    }
}
