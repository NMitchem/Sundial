# df64 accumulator experiment (M2 Task 5)

## Hypothesis

Double-double (two-f32, "df64") accumulation in the accumulation-critical GPU kernels
(CSR SpMV, transport row/col sums, dot/sum reductions) should recover most of the
precision that plain f32 loses on cancellation-heavy inputs — at ~roughly f32 cost, since
only the *accumulators* are df64 while all buffers stay f32. Behind `SolveOptions::df64`
/ CLI `--df64`, default off.

A df64 value is `vec2<f32>(hi, lo)` with `|lo| <= ulp(hi)/2`. It rests on two error-free
transforms that capture the rounding error of a single f32 op:

- `two_prod(a,b)`: `p = a*b; e = fma(a, b, -p)` — `e` is the exact product tail.
- `two_sum(a,b)`: `s = a+b; bb = s-a; err = (a-(s-bb)) + (b-bb)` — `err` is the exact sum tail.

These tails are non-zero *only because of f32 rounding*; in exact real arithmetic they are 0.

## Result: refuted on the wgpu-30 Metal backend (Apple M4 Pro)

Adversarial cancellation dot product, n = 1,000,000, seed 42, log-uniform magnitudes
1e-3..1e6, random signs (see `crates/sundial-core/tests/gpu_df64.rs::adversarial_pair`):

| quantity | value |
|---|---|
| f64 ground truth | 1.809929e8 |
| f32 error (Neumaier-compensated reduce) | 7.552e0 |
| df64 error (double-double reduce) | 7.552e0 |

**df64 delivered zero additional precision** — the two errors are identical.

### Root cause

Metal compiles with fast-math on, which treats float ops as exact/associative. The
compiler then proves `fma(a,b,-a*b) == 0` and `(a-(s-bb))+(b-bb) == 0` and folds every
error term to zero, collapsing df64 to plain f32. (It also collapses the f32 path's
Neumaier compensation to zero for the same reason, so both paths degrade to the same
plain-f32 accumulation — which is why the two errors come out bit-identical.)

### Why there is no fix in wgpu 30 (source-verified, pinned `wgpu = 30.0.0`, `naga = 30.0.0`)

1. **`objc2-metal` 0.3.2** — `MTLCompileOptions.fastMathEnabled` documented default: **YES**
   ("optimizations for floating-point arithmetic that may violate the IEEE 754 standard").
2. **wgpu-hal 30 `src/metal/device.rs`** — the only shader-compile site creates
   `MTLCompileOptions::new()` and sets **only** `setLanguageVersion` + `setPreserveInvariance(true)`.
   It never sets `mathMode`/`fastMathEnabled`, so fast-math stays at its YES default.
   (`preserveInvariance` governs cross-invocation result stability, not fast-math.)
3. **naga 30 `back::msl::Options`** — no math-mode / precise / contract field; the writer
   emits plain `*`/`+`/`-` and `fma`, which Metal then contracts.
4. **wgpu 30 / wgpu-types 30 public API** — no fast-math / math-mode / precise control on
   `DeviceDescriptor`, `ShaderModuleDescriptor`, or features.

WGSL has no `volatile` / optimization barrier to protect the intermediates from inside the
shader either. So on this stack there is no supported way to keep the error-free transforms alive.

## What still holds

- The df64 kernels are **numerically correct** (just not more precise on Metal): they run,
  bind correctly (including the transport `ot_apply_df64` 3/4/7 remap), and produce
  verified-Optimal solutions. `solve afiro --df64` → Optimal obj -464.7530938908 (mu 7.18e-7);
  `transport --grid 16 --df64` → Optimal obj 0.0879941516 (mu 4.81e-5).
- Default-off is byte-identical to the plain-f32 tree.

## To actually measure the df64 gain

Re-run the experiment on a backend where fast-math can be disabled / is off — e.g. a
Vulkan path (native or MoltenVK) with strict math, or a future wgpu that surfaces a
math-mode control. On such a backend `gpu_df64::df64_dot_accuracy_measured` would print a
df64 error orders of magnitude below the f32 error. As of this stack, that gain is
unobservable on Apple/Metal; do not cite df64 accuracy numbers taken on Metal.
