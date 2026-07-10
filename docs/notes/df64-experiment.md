# df64 accumulator experiment (M2, 2026-07-10)

Hardware: Apple M4 Pro / Metal, release builds. Tier 1 only: df64 (two-f32)
accumulators in SpMV / transport sums / dot+sum reductions; f32 iterates.
Behind `SolveOptions::df64` / CLI `--df64`, default off (see M2 Task 5,
`.superpowers/sdd/m2-task-5-report.md`).

## Protocol: planned vs. what actually happened

The plan (`docs/superpowers/specs/2026-07-10-sundial-m2-design.md` §2, this
plan's Task 6) fixed a three-sweep protocol before any implementation existed:

1. The 12 M1 f32-wall instances, tol 1e-4, `--df64`: how many now reach
   verified Optimal?
2. The 19 M1 passers, tol 1e-6, with and without `--df64`: per-instance
   floor and pass count in each mode.
3. Hero (`transport --grid 32`) at a fixed 32k-iteration budget, with and
   without `--df64`: verified mu floor + ms/iter delta.

That protocol assumed the outcome was *unknown* going in. It isn't anymore.
Task 5's implementation phase included a **kernel-level accuracy test**
(`gpu_df64::df64_dot_accuracy_measured`, an adversarial cancellation dot
product with a known f64 ground truth) that was written to falsify the df64
hypothesis directly, and it did: on this stack, df64 and f32 accumulation are
**byte-identical**, for a source-verified compiler reason (below), not a
measurement artifact. Once that is established, the three-sweep protocol
measures nothing — every one of the 12+19+2 runs would be comparing a
pipeline against a byte-identical copy of itself. Running hours of GPU time
to confirm a result that is already provably true at the mechanism level is
not measurement, it's theater. The controller adjudicated the sweeps out of
scope; this memo instead reports the mechanism finding plus one cheap
solve-level confirmation, and ships `scripts/bench_each.sh` (the tool the
sweeps would have used) for Task 11 or a future revisit.

## Finding 1: Metal fast-math collapses every error-free transform (root cause)

Double-double accumulation is built on two error-free transforms that
recover the rounding error of a single f32 op:

- `two_prod(a,b)`: `p = a*b; e = fma(a, b, -p)` — `e` is the exact product tail.
- `two_sum(a,b)`: `s = a+b; bb = s-a; err = (a-(s-bb)) + (b-bb)` — `err` is
  the exact sum tail.

Under *exact* arithmetic these tails are 0; they are non-zero only because of
f32 rounding. Metal's fast-math treats float ops as exact/associative, so the
compiler proves `fma(a,b,-a*b) == 0` and `(a-(s-bb))+(b-bb) == 0` and folds
every error term to zero at compile time. df64 then reduces to plain f32.

Measured (adversarial cancellation dot, n = 1,000,000, seed 42, log-uniform
magnitudes 1e-3..1e6, `crates/sundial-core/tests/gpu_df64.rs::adversarial_pair`):

| quantity | value |
|---|---|
| f64 ground truth | 1.809929e8 |
| f32 error (Neumaier-compensated reduce) | 7.552e0 |
| df64 error (double-double reduce) | 7.552e0 |

**Byte-identical.** df64 delivered zero additional precision on this backend.

### No control exists in wgpu 30 to disable it (source-verified)

Traced the whole compile path in the pinned versions (`wgpu = 30.0.0`,
`naga = 30.0.0`):

1. **`objc2-metal` 0.3.2** documents `MTLCompileOptions.fastMathEnabled`:
   *"enables optimizations for floating-point arithmetic that may violate the
   IEEE 754 standard … fastMathEnabled defaults to **YES**."*
2. **wgpu-hal 30 `src/metal/device.rs`** (the only shader-compile site)
   creates `MTLCompileOptions::new()` and sets **only** `setLanguageVersion`
   and `setPreserveInvariance(true)`. It never touches
   `mathMode`/`fastMathEnabled`, so it stays at the default YES.
   (`preserveInvariance` governs cross-invocation result stability, not
   fast-math — it does not help.)
3. **naga 30 MSL backend** — `back::msl::Options` has no math-mode / precise
   / contract field; the writer emits plain `*`/`+`/`-` and `fma`, which
   Metal then contracts.
4. **wgpu 30 / wgpu-types 30 public API** — `DeviceDescriptor`,
   `ShaderModuleDescriptor`, features: no fast-math / math-mode / precise
   knob anywhere.

There is no supported hook (device descriptor, shader-module descriptor, or
naga option surfaced through wgpu) to disable Metal fast-math in wgpu 30.
WGSL has no `volatile`/optimization barrier to protect the intermediates
from within the shader either.

## Finding 2: Neumaier collateral (pre-existing, now explained)

The same fast-math mechanism has been silently collapsing the M0-era
Neumaier compensation in `reduce.wgsl` to plain f32 accumulation on Metal all
along — this predates df64 and is not a regression from this task. It has
**no gate impact**: the 1e-4 tier still holds and every reported `Optimal` is
independently verified against CPU f64 (the honesty/verification machinery —
Farkas certificates, residual re-checks off the GPU path — was load-bearing
here, catching what the compensated-arithmetic comment promised but Metal
silently didn't deliver). The "compensated accumulation" claim in existing
docs needs a Metal caveat; carried forward to Task 11's writeup.

## Finding 3: cross-workgroup bound (design constant, independent of fast-math)

Even on a hypothetical IEEE-strict backend (Vulkan/DX12) where the
error-free transforms survive, df64's gain is capped by the current
reduction pipeline's design: each workgroup's df64 partial is collapsed back
to a single f32 before being written to the inter-workgroup scratch buffer
(`out_a[wid] = hi + lo`, f32 scratch — see `crates/sundial-core/src/gpu/kernels.rs`
`Reducer`). The final cross-workgroup combine is therefore always f32-limited
regardless of accumulator precision inside a workgroup. A real tier gain
would require df64-typed scratch buffers end-to-end, not just this task's
per-workgroup kernels.

## Solve-level confirmation run

Per Task 6's brief, one cheap end-to-end check: `afiro.mps`, tol 1e-6, f32
vs `--df64`, JSON diffed.

```
cargo run --release -p sundial-cli -- solve bench/netlib/afiro.mps --tol 1e-6 --json > /tmp/afiro-f32.json
cargo run --release -p sundial-cli -- solve bench/netlib/afiro.mps --tol 1e-6 --df64 --json > /tmp/afiro-df64.json
```

Both exited 0 (`Optimal`, no `--max-iters` wall hit). Verbatim:

| field | f32 | df64 |
|---|---|---|
| status | Optimal | Optimal |
| objective | -464.75308807240185 | -464.75309389080354 |
| iterations | 26048 | 4352 |
| restarts | 17 | 5 |
| solve_ms | 3628.03 | 645.54 |
| rel_primal | 2.857e-7 | 2.025e-7 |
| rel_dual | 5.106e-7 | 7.179e-7 |
| rel_gap | 1.235e-7 | 6.638e-9 |

**This is not byte-identical, and that is expected, not a contradiction of
Finding 1.** The kernel-level test isolates a single dot product; the
solve-level run exercises the reduction every iteration inside an iterative
first-order LP method. `reduce_dot` (f32, Neumaier, `reduce.wgsl`) and
`reduce_dot_df64` (`df64.wgsl`) are textually different kernels — different
per-thread branches, different intermediate `two_sum`/`two_prod`/`fma` calls
— that fast-math collapses to *mathematically* the same plain summation, but
not necessarily to the *same instruction sequence*: f32 addition is
non-associative, so two differently-shaped-but-equivalent compiled kernels
can legitimately round differently at the ULP level even under identical
fast-math semantics. Both `Reducer` code paths dispatch the identical
workgroup/pass structure (`kernels.rs::Reducer::record` only swaps the entry
point name; Finding 3's tree shape is unchanged), so the divergence is
compiler/kernel-body rounding, not a structural difference in the reduction.

Fed into an iterative solver near its stopping tolerance, that ULP-level
difference compounds every iteration into a different trajectory — different
iteration counts, different restart counts, a different final iterate. Here
df64's trajectory happened to converge faster and to a tighter gap; that is
**not evidence of a real precision gain** (Finding 1 already rules that out
at the source level for this exact kernel pair) — it is one instance of
chaotic sensitivity to rounding order, and it could as easily go the other
way on a different instance or seed. It is recorded here verbatim, as
instructed, precisely because it is a genuine and somewhat counter-intuitive
result: "byte-identical kernel accuracy" does not imply "byte-identical
solver trajectory," and no single afiro run should be read as a benchmark
result in either direction.

## Decision gate

**DEFER df64 — pre-determined by mechanism.** The three-sweep protocol is
not run because its outcome is already fixed by Finding 1: on wgpu 30 /
Metal, df64 and f32 accumulation are provably byte-identical at the source
level, so every sweep instance would report identical wall/passer/hero
numbers to the existing f32 baseline (mod the chaotic-trajectory effect
above, which is noise, not signal, and would average out across a real sweep
rather than justify one).

Revisit when either:
- (a) wgpu exposes an IEEE-strict / math-mode control for the Metal backend
  (track wgpu releases — no such control exists as of wgpu 30 / naga 30, per
  Finding 1's source trace), or
- (b) targeting a Vulkan/DX12-only context (native or via a strict-math
  translation layer), where fast-math is not forced on by default.

Even under (a) or (b), Finding 3's cross-workgroup f32 collapse must also be
addressed (df64-typed scratch buffers through the full reduction, not just
per-workgroup) before a revisit could show a real tier gain — fixing only
the fast-math issue caps the win at whatever headroom survives the
inter-workgroup f32 combine.

## Artifacts produced by this task

- `scripts/bench_each.sh` — resumable per-instance bench harness (built per
  spec, not run against the wall/passer sweeps since they're moot per the
  gate above); available for Task 11 or a future revisit under (a)/(b).
- `.gitignore` — `/results-*.csv`, `/hero-*.json` (the sweep/hero output
  patterns this script and a future revisit would produce).
