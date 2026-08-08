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

var<workgroup> sh: array<f32, 256>;

fn wg_tree_reduce_sum(lid: u32) {
    workgroupBarrier();
    var s = 128u;
    loop {
        if (lid < s) { sh[lid] = sh[lid] + sh[lid + s]; }
        workgroupBarrier();
        s = s >> 1u;
        if (s == 0u) { break; }
    }
}

// partial dot(a,b) with Neumaier compensation per thread
// in_a=a  in_b=b  out_a=partials[num_workgroups]   (P.n = len, P.stride = total threads)
@compute @workgroup_size(256)
fn reduce_dot(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    var sum = 0.0; var comp = 0.0;
    var i = gid.x;
    while (i < P.n) {
        let v = in_a[i] * in_b[i];
        let t = sum + v;
        if (abs(sum) >= abs(v)) { comp = comp + ((sum - t) + v); }
        else { comp = comp + ((v - t) + sum); }
        sum = t;
        i = i + P.stride;
    }
    sh[lid.x] = sum + comp;
    wg_tree_reduce_sum(lid.x);
    if (lid.x == 0u) { out_a[wid.x] = sh[0]; }
}

// partial sum(a) — same structure, v = in_a[i]
@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) gid: vec3<u32>,
              @builtin(local_invocation_id) lid: vec3<u32>,
              @builtin(workgroup_id) wid: vec3<u32>) {
    var sum = 0.0; var comp = 0.0;
    var i = gid.x;
    while (i < P.n) {
        let v = in_a[i];
        let t = sum + v;
        if (abs(sum) >= abs(v)) { comp = comp + ((sum - t) + v); }
        else { comp = comp + ((v - t) + sum); }
        sum = t;
        i = i + P.stride;
    }
    sh[lid.x] = sum + comp;
    wg_tree_reduce_sum(lid.x);
    if (lid.x == 0u) { out_a[wid.x] = sh[0]; }
}

// partial sum((a-b)^2) with Neumaier compensation per thread — the squared L2
// movement ‖a−b‖² for the movement-based primal weight. Differencing INSIDE the
// kernel matters: at a restart a and b are neighbouring iterates, so a separate
// subtract-then-dot pass would round the difference to f32 before squaring it
// and lose the small-movement regime the weight update depends on.
// in_a=a  in_b=b  out_a=partials[num_workgroups]
@compute @workgroup_size(256)
fn reduce_diff_sq(@builtin(global_invocation_id) gid: vec3<u32>,
                  @builtin(local_invocation_id) lid: vec3<u32>,
                  @builtin(workgroup_id) wid: vec3<u32>) {
    var sum = 0.0; var comp = 0.0;
    var i = gid.x;
    while (i < P.n) {
        let d = in_a[i] - in_b[i];
        let v = d * d;
        let t = sum + v;
        if (abs(sum) >= abs(v)) { comp = comp + ((sum - t) + v); }
        else { comp = comp + ((v - t) + sum); }
        sum = t;
        i = i + P.stride;
    }
    sh[lid.x] = sum + comp;
    wg_tree_reduce_sum(lid.x);
    if (lid.x == 0u) { out_a[wid.x] = sh[0]; }
}

// partial max|a|
@compute @workgroup_size(256)
fn reduce_maxabs(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>,
                 @builtin(workgroup_id) wid: vec3<u32>) {
    var mx = 0.0;
    var i = gid.x;
    while (i < P.n) {
        mx = max(mx, abs(in_a[i]));
        i = i + P.stride;
    }
    sh[lid.x] = mx;
    workgroupBarrier();
    var s = 128u;
    loop {
        if (lid.x < s) { sh[lid.x] = max(sh[lid.x], sh[lid.x + s]); }
        workgroupBarrier();
        s = s >> 1u;
        if (s == 0u) { break; }
    }
    if (lid.x == 0u) { out_a[wid.x] = sh[0]; }
}
