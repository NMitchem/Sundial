# I solved a 1,048,576-variable optimization problem in a browser tab, on any GPU

*Show HN draft — Sundial, a linear-programming solver that runs as WebGPU compute shaders.*

Live demo: <DEMO_URL>

---

## 1. The hook

Open a web page. Pick a grid size. Watch a 1,048,576-variable linear program
converge to an optimal solution on your own laptop's GPU — no install, no
sign-up, no server doing the work for you, and no NVIDIA card required. It
runs on Apple, AMD, Intel, and NVIDIA alike, because it's built on WebGPU
compute shaders instead of CUDA.

The demo problem is optimal transport: move the mass of one 32×32 image onto
another as cheaply as possible. That's 1,024 sources × 1,024 sinks =
1,048,576 transport variables and 2,048 marginal constraints. On an Apple M4
Pro it solves to a verified 1e-4 tolerance in about 9.4 seconds natively, and
in the same ballpark inside a browser tab. Every "Optimal" you see was
re-checked in double precision on the CPU before the word was allowed to
appear — more on that below, because it's the part that matters.

## 2. Why this didn't exist yet

Linear programming had a GPU moment over the last two years. First-order
methods — PDLP from Google, then cuPDLP, then NVIDIA's cuOpt, and as of April
2026 even HiGHS's new GPU path ("HiPDLP") — showed that you can solve very
large LPs on a GPU by doing nothing but sparse matrix-vector products and
vector reductions, no matrix factorization anywhere. The catch: every one of
those implementations is CUDA. cuPDLP-C is CUDA-only (it even ships an
official single-precision `SFLOAT` build flag, which is what told us f32 was a
supported mode and not a research gamble). HiGHS's GPU path "runs on an NVIDIA
GPU," Linux and Windows only. MPAX is vendor-portable but it's a Python/XLA
install. The lock-in has been *deepening*, not resolving.

Meanwhile, the solvers that *do* run in a browser — clp-wasm, highs-js, YALPS
— are all CPU WASM ports of simplex-era code. Google's own OR-Tools WASM port
documents that its GPU-dependent code doesn't survive the port to the browser.

So there was a gap with nobody in it: GPU-parallel LP × any GPU vendor ×
zero-install browser execution. Two independent prior-art sweeps ("WebGPU
linear programming," "WebGPU PDHG," "WGSL optimization solver," GitHub topic
crosses, arXiv, HN) turned up zero hits. WebGPU is what changes the
constraint: it's the first portable GPU-compute API that ships in browsers and
runs natively through the same `wgpu` stack. Sundial is restarted PDHG (the
PDLP algorithm family) written in f32 WGSL, and it runs unmodified both
natively and in the tab.

## 3. The honesty machinery (the part that matters)

A GPU LP solver in f32 has an obvious credibility problem: single precision is
noisy, first-order methods crawl toward their tolerance, and it's very easy to
declare victory a little early. Sundial's answer is a rule we never break:
**the GPU never grades its own homework.**

The GPU iterates in f32 and *flags* when it thinks it's done. But `Optimal` is
only ever set after a completely separate check: the returned primal-dual
point is re-evaluated against the full KKT conditions in f64 on the CPU, and
the returned solution *is* that verified point. The duality gap in that check
is evaluated at a sign-projected dual, so on standard-form problems the gap is
genuinely enforced rather than silently dropped (an unprojected multiplier can
send an f64 term to ±infinity and quietly turn the gap into a NaN that falls
out of the certificate — we hit exactly that and fixed it).

M2 extended the same discipline to the two other things an LP can be. Sundial
now detects **Infeasible** and **Unbounded** — but only after a CPU-f64 Farkas
certificate (a dual ray for infeasibility, a primal recession ray for
unboundedness) verifies it. The GPU engine only nominates a candidate when the
iterate norm keeps growing across restarts; the f64 certificate is the sole
authority that gets to set the status. A false "Infeasible" is therefore
structurally impossible in the same way a false "Optimal" is.

Getting the *nomination* heuristic right had a nice subtlety. Our first cut
watched for *geometric* growth in the iterate norm. It never fired — and there
was a reason. PDHG divergence is *linear*: the iterate difference converges to
a fixed ray (Applegate et al.), so the ratio between successive restarts
decays toward 1 and a geometric threshold can't sustain a streak by
construction. We switched to a monotonic-with-margin test (each restart's norm
at least 1.02× the last), which is just a noise margin, not a growth rate.
Because the f64 certificate is still the only thing that can set a status, this
constant can only ever change *when* we attempt verification, never *whether* a
wrong answer slips through. On constructed infeasible/unbounded oracles the
certificate fires at about 12.4k iterations.

In the field, the detector is deliberately conservative, and the honest
recall number reflects that. On Netlib's real infeasible set, **2 of 6
instances certify Infeasible** (itest2 and galenet); the other **4 stop at an
honest `IterationLimit`**, and — the number that actually matters — **zero
produced a false `Optimal`**. That trade is on purpose: a missed detection
costs you some iterations, a false claim would cost you trust, and the whole
architecture is built so the false-claim direction is structurally impossible.
We'd rather report "I couldn't decide" than "solved" when we can't prove it.

## 4. The war story: the million-variable gate that stalled at half a million iterations

The 1M-variable transport problem did not work the first time, and the way it
failed is the most instructive thing in the project.

With the textbook PDHG step sizes (τ = σ), the solver ran to its iteration cap
— 500,032 iterations, about 290 seconds — and stopped at a verified mu of
1.23e-4, just past the 1e-4 line. Reading the residual curves explained why:
the **dual** residual had collapsed to about 1e-9, essentially converged,
while the **primal** residual sat on a plateau around 1e-4 and would not come
down. The two sides of the problem were progressing at wildly different rates
and the single shared step size couldn't serve both.

The fix is PDLP's primal-weight balancing: introduce a weight ω that
rebalances the primal and dual step sizes, initialize it from the ratio of the
cost and constraint norms in iterate space, and nudge it at each restart to
equalize the two residuals (a √-damped residual-balance update, ω clamped to
[1e-4, 1e4]). With ω in place the same problem converges in **16,000
iterations, 9 restarts, about 9.4 seconds**, verified mu 9.83e-5.

One honest wrinkle we wrote down rather than hid: ω only helps on the
matrix-free/unscaled path. On the explicit path (Ruiz + preconditioner
equilibration already applied), the residual-ratio update limit-cycled between
about 0.02 and 0.12 and actually regressed one instance, so there we kept the
original τ = σ behavior, which we proved bit-identical to the pre-ω code.
Movement-based weighting for that path is on the backlog. The screenshots in
the demo reproduce the good curve live: dual dropping fast, primal following
once ω kicks in.

## 5. What the browser costs you

Almost nothing, at this scale. The native run does about 0.59 ms per iteration
on the M4 Pro. The same code compiled to wasm and driven through the browser's
WebGPU does about 0.547 ms per iteration — within measurement noise of native,
which is to say the browser is effectively free here. The 32×32 (1,048,576
variable) problem reaches `Optimal (CPU f64 verified)` in the tab at 16,000
iterations; the 16×16 problem (65,536 variables) lands in about one second.

That result is not a foregone conclusion — WebGPU has storage-buffer size
limits and driver quirks, and a fair amount of the engineering was staying
inside the portable WGSL subset (no subgroups, no f16/f64, ±1e30 sentinels
instead of infinity arithmetic so nothing depends on a browser honoring IEEE
inf). But once you're inside that subset, the GPU is the GPU, tab or no tab.

## 6. The double-double experiment, and why it's a *negative* result

The original pitch promised a "double-double f32" trick — emulate ~46 bits of
mantissa by carrying each number as a hi/lo pair of f32s — to push past the
1e-4 tier. We built it (behind a `--df64` flag, default off) and then wrote a
kernel-level test designed to *falsify* it: an adversarial cancellation dot
product with a known f64 ground truth. It falsified it. On Apple/Metal, df64
and plain f32 accumulation came out **byte-identical** — same error, 7.552e0,
to the digit.

The reason is not a bug in our code; it's the compiler, and we traced it to
the source. Double-double is built on "error-free transforms" like
`two_prod(a,b): p = a*b; e = fma(a, b, -p)`, where `e` is the exact rounding
tail of the product. Metal's fast-math is on by default and treats float ops
as exact/associative, so it proves `fma(a, b, -a*b) == 0` and folds every
error term to zero at compile time — the double-double collapses back to f32
before it ever runs. And there is no switch to turn it off: we read through
wgpu-hal's Metal device code, naga's MSL backend, and the whole public wgpu 30
surface — none of them expose a fast-math / precise / math-mode control, and
`MTLCompileOptions.fastMathEnabled` defaults to YES.

Two consequences worth stating plainly. First, the *existing* "compensated
accumulation" (a Neumaier sum in the M0 reduction kernel) has been collapsing
to plain f32 on Metal since day one, for the same reason — which had zero
impact on results *precisely because* the CPU-f64 certificate was the real
guarantee all along, catching what the comment in the shader promised but the
hardware quietly didn't deliver. Second, even on an IEEE-strict backend where
the transforms survive, our current reduction still narrows each workgroup's
double-double partial back to a single f32 before the cross-workgroup combine,
so a real precision win would need df64-typed scratch buffers end to end, not
just df64 math inside a kernel. So: **df64 is deferred**, with a written,
source-level reason and a concrete list of what would have to change to revisit
it. (We do not claim the one afiro solve where f32 and df64 finished with
different iteration counts as a df64 "win" — that's chaotic sensitivity to
rounding order in an iterative method, noise in both directions, not a
precision gain. The memo explains why.)

That's the honest version of "the double-double trick alone is a good post":
the post is about why it *doesn't* work on the most popular WebGPU backend, and
how you can prove that from the compiler down rather than guessing.

## 7. Benchmarks

Sundial runs the classic Netlib LP set through a CLI sweep — one row per
instance, each solved on the GPU and then CPU-f64-verified, reported against
the published Netlib optima. The current split on 32 instances:

**20 of 32 reach Optimal** at CPU-f64-verified 1e-4; the other **12 stop at
`IterationLimit`**, and there are **no parse failures** (down from one in M1 —
see `blend.mps` below). Among the 20 Optimal rows, the worst relative objective
error (reported as |obj − known| / (1 + |known|)) against the readme is e226's
3.6e-1 — which is the sign-convention footnote below, not a real error; every
other Optimal instance matches the published optimum to better than 1e-3 (worst
real case: adlittle, 6.7e-4). This
is up from M1's 19/32 Optimal: the one instance that changed is `blend.mps`,
which went from an unreadable parse error to a verified solve.

`blend.mps` is new this milestone: it used to fail the parser on a
set-name-less RHS line (a real-world MPS corner), and now it parses and solves
to Optimal (−30.8119660669 against the optima-file −30.812149846, gap within
the verified mu of 9.98e-5). The instances that don't reach Optimal stop
honestly at the iteration cap — that's the documented f32 wall, reported as
`IterationLimit`, never dressed up as a solve.

One footnote the report renders itself: **e226**. Its Netlib-readme "known
optimum" uses the opposite sign convention for the objective-row RHS constant,
so our KKT-certified −11.635074 shows a large relative error against the
readme's ≈ −18.75. That's a convention mismatch, not a solver defect; the
report annotates it in a note column rather than quietly "fixing" the number.

**On comparing against published GPU-LP benchmarks:** we looked for overlap
with the Mittelmann benchmarks (plato.asu.edu/bench.html, the standard
reference, which does include GPU solvers cuOpt and cuPDLPx). There is none to
cite honestly. Those benchmarks target data-center-scale instances — the small
end of their GPU feasibility set (qap15, ~6,331 rows × ~22,275 columns) already
dwarfs every instance in our classic-Netlib sweep, the large end runs to tens
of millions of rows and columns, and the GPU solvers there (cuOpt, cuPDLPx) run
on an NVIDIA B200. None of our 32 instances appear in them, and they publish
results at different tolerance tiers on different-class hardware. So rather than manufacture a misleading
apples-to-oranges table, we state our absolute numbers above and point you at
the plato page: <https://plato.asu.edu/bench.html>. The comparison that *is*
honest and reproducible is the one anyone can run — open the demo, solve a
million-variable problem, and time it on your own GPU.

## 8. Honest limits

- **f32 iterates, 1e-4 default tier.** The iterate arithmetic is single
  precision and the headline tolerance is PDLP's "moderate accuracy" 1e-4 tier.
  Tighter tolerances are genuinely future work — and per Section 6, the obvious
  double-double route is blocked on Apple/Metal until wgpu exposes a strict-math
  control (or you target a Vulkan/DX12 context, and even then the reduction
  pipeline needs df64 scratch buffers).
- **Not every Netlib instance reaches Optimal in f32.** The `IterationLimit`
  rows in the table are the f32 wall, stated as such. They are honest
  non-solves, not silent failures.
- **No presolve.** We evaluated the recent GPU-presolve work (Cederberg & Boyd,
  arXiv 2604.23951) and deferred it: it only applies to the explicit-CSR path
  (not the matrix-free transport path), its wins concentrate in a minority of
  large/redundant instances, and the postsolve mapping that keeps certificate
  honesty intact is a multi-week correctness-critical effort. Deferred to M3+,
  with the reasoning written down.
- **Metal fast-math caveat.** Any "compensated accumulation" is collapsed to
  plain f32 on the Metal backend (Section 6). The CPU-f64 certificate is what
  makes this safe; don't read the shader comments as a precision guarantee.
- **Small LPs still belong to simplex on a CPU.** A first-order GPU method pays
  off at scale; the million-variable transport hero is where the architecture
  earns its keep, not afiro. Simplex will beat us on the little ones and that's
  expected.
- **WebGPU availability floor.** You need a browser with WebGPU: Chrome/Edge
  113+, Firefox 141+, Safari 26+. Timing convention: reported `solve_ms` is the
  solve loop only, excluding host preprocessing (Ruiz scaling + power-iteration
  norm), consistent across the CPU and GPU engines.

## 9. Try it / use it

**In the browser** — nothing to install:

> <DEMO_URL>

Pick a preset (blobs, ring, spiral, checker, corners) or draw your own source
and target masses with the brush; choose a 16×16 or 32×32 grid; watch the three
live heatmaps (source, mass arriving, target) and the convergence chart. There's
also a drop-a-file benchmark page: hand it an `.mps` or `.mps.gz` and get an
honest results table back.

**As a library** — the solver ships as an npm/WASM package:

```bash
npm install sundial-lp
```

```js
import init, { solveMps } from "sundial-lp";
await init();                                    // load the wasm module

// solveMps(mpsText, tol, onProgress) runs on the browser's GPU and
// resolves once the result has been re-verified in f64 on the CPU.
const result = await solveMps(mpsText, 1e-4, (p) => {
  // p = { iter, rel_primal, rel_dual, rel_gap, ms_per_iter } each check
});
// result = { status, objective, iterations, ... }
```

Private data never leaves the browser — the optimization runs client-side on
the user's own GPU, with no backend.

**On the command line** — native Metal/Vulkan/DX12 via `wgpu`:

```bash
# solve an MPS file
cargo run -p sundial-cli --release -- solve path/to/model.mps

# the 1M-variable transport hero (two 32x32 grids, ~9.4s on an M4 Pro)
cargo run -p sundial-cli --release -- transport --grid 32

# sweep a directory of instances into a report
cargo run -p sundial-cli --release -- bench bench/netlib --out results.csv
cargo run -p sundial-cli --release -- report results.csv --out report.md
```

Source is Rust + `wgpu` + WGSL, MIT OR Apache-2.0. The whole thing is one
codebase that runs on the CPU (f64 reference), on a native GPU, and in a
browser tab — and it re-verifies every optimum in double precision before it
tells you it's optimal.
