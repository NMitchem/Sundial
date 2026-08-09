# I solved a 1,048,576-variable optimization problem in a browser tab, on any GPU

*Show HN draft for Sundial, a linear-programming solver that runs as WebGPU compute shaders.*

Live demo: <DEMO_URL>

---

Every GPU linear-programming solver shipped in the last two years runs on CUDA.
cuPDLP is CUDA-only. NVIDIA's cuOpt is CUDA. HiGHS shipped a GPU path in April
2026, and it runs on an NVIDIA GPU, Linux and Windows. Meanwhile every LP
solver that runs in a browser (clp-wasm, highs-js, YALPS) is a CPU port of
simplex-era code, and Google's own OR-Tools WASM port documents that its
GPU-dependent code doesn't survive the trip.

So the two lists never overlap.

Sundial is 1,048,576 variables in **9.4 seconds**, on an Apple laptop, in a tab.

## The gap nobody was standing in

Linear programming had a GPU moment over the last two years. First-order
methods showed you can solve very large LPs with nothing but sparse
matrix-vector products and vector reductions, no matrix factorization anywhere.
PDLP came out of Google, then cuPDLP, then cuOpt, then HiPDLP. The math is
settled and public.

The distribution isn't. cuPDLP-C even ships an official single-precision
`SFLOAT` build flag, which is how we knew f32 was a supported mode and not a
research gamble. It's still CUDA. MPAX is vendor-portable, and it's a
Python/XLA install. The lock-in has been deepening, not resolving.

Two independent prior-art sweeps ("WebGPU linear programming," "WebGPU PDHG,"
"WGSL optimization solver," GitHub topic crosses, arXiv, HN) turned up zero
hits on the intersection: GPU-parallel LP, any GPU vendor, zero-install browser
execution.

WebGPU is what moved. It's the first portable GPU-compute API that ships in
browsers and runs natively through the same `wgpu` stack, so one codebase
covers Metal, Vulkan, DX12, and the tab. Sundial is restarted PDHG in f32 WGSL,
and it runs unmodified in all four.

## The GPU never grades its own homework

An f32 GPU solver has an obvious credibility problem. Single precision is
noisy, first-order methods crawl toward their tolerance, and declaring victory
a little early is easy and completely invisible from the outside.

So there's one rule. The GPU iterates and the GPU nominates. It never sets a
status.

`Optimal` is only ever set after a separate check: the returned primal-dual
point is re-evaluated against the full KKT conditions in f64 on the CPU, and
the solution you get back *is* that verified point. The duality gap in that
check is evaluated at a sign-projected dual, so on standard-form problems the
gap is genuinely enforced rather than silently dropped. That last detail is not
theoretical. An unprojected multiplier sends an f64 term to infinity and turns
the gap into a NaN that falls out of the certificate entirely. We hit exactly
that, and fixed it.

The same discipline covers the other two things an LP can be. `Infeasible` and
`Unbounded` are set only after a CPU-f64 Farkas certificate verifies them: a
dual ray for infeasibility, a primal recession ray for unboundedness. The GPU
engine nominates a candidate when the iterate norm keeps growing across
restarts. The f64 certificate is the only authority that gets to set the
status, so a false `Infeasible` is structurally impossible in the same way a
false `Optimal` is.

Getting the nomination heuristic right had a nice subtlety. The first cut
watched for *geometric* growth in the iterate norm, and it never fired once.
There's a reason for that. PDHG divergence is linear: the iterate difference
converges to a fixed ray (Applegate et al.), so the ratio between successive
restarts decays toward 1 and a geometric threshold can't sustain a streak by
construction. We switched to monotonic-with-margin, where each restart's norm
has to be at least 1.02× the last. That's a noise margin, not a growth rate.
Because the f64 certificate is still the only thing that can set a status, this
constant can only change *when* we attempt verification, never *whether* a
wrong answer gets through. On constructed oracles the certificate fires at
about 12,400 iterations.

In the field the detector is deliberately conservative, and the recall number
says so. On Netlib's real infeasible set, **2 of 6 instances certify**
(itest2 and galenet). The other **4 stop at an honest `IterationLimit`**. And
the number that actually matters: **zero produced a false `Optimal`**.

That trade is on purpose. A missed detection costs you iterations. A false
claim costs you trust, and the whole architecture is built so the false-claim
direction can't happen. We'd rather report "I couldn't decide" than "solved."

## The million-variable gate that stalled at half a million iterations

The 1M-variable transport problem did not work the first time, and how it
failed is the most instructive thing in the project.

With textbook PDHG step sizes (τ = σ), the solver ran to its iteration cap:
500,032 iterations, about 290 seconds, stopping at a verified µ of 1.23e-4.
Just past the 1e-4 line. Maddening.

The residual curves explained it. The **dual** residual had collapsed to about
1e-9, essentially converged. The **primal** residual sat on a plateau around
1e-4 and would not come down. Two sides of the same problem, progressing at
wildly different rates, sharing one step size that could only serve one of
them.

The fix is PDLP's primal-weight balancing. Introduce a weight ω that rebalances
the primal and dual step sizes, initialize it from the ratio of the cost and
constraint norms in iterate space, and nudge it at each restart to equalize the
two residuals. With ω in place the same problem converges in **16,000
iterations, 9 restarts, about 9.4 seconds**, at a verified µ of 9.83e-5.

That's 31× fewer iterations from a change that touches two scalars.

One wrinkle we wrote down instead of hiding: ω only helps on the
matrix-free/unscaled path. On the explicit path, where Ruiz and Pock–Chambolle
equilibration already ran, the residual-ratio update limit-cycled between about
0.02 and 0.12 and regressed an instance. So that path keeps τ = σ, which we
proved bit-identical to the pre-ω code. Section 7 is what eventually came of
that gap.

## The browser is free

Native runs at about 0.59 ms per iteration on the M4 Pro. The same code
compiled to wasm and driven through the browser's WebGPU runs at about
**0.547 ms**. That's inside measurement noise, which is to say the tab costs
you nothing at this scale. The 32×32 problem reaches `Optimal (CPU f64
verified)` in the browser at 16,000 iterations. The 16×16 problem, 65,536
variables, lands in about a second.

That wasn't a foregone conclusion. WebGPU has storage-buffer size limits and
driver quirks, and a real share of the engineering went into staying inside the
portable WGSL subset: no subgroups, no f16, no f64, and ±1e30 sentinels instead
of infinity arithmetic so nothing depends on a browser honoring IEEE inf. Once
you're inside that subset, though, the GPU is the GPU. Tab or no tab.

## The trick that didn't work, and why that's the interesting part

The original pitch promised a double-double trick: carry each number as a hi/lo
pair of f32s, emulate about 46 bits of mantissa, push past the 1e-4 tier. We
built it behind a `--df64` flag, then wrote a kernel-level test designed to
falsify it. An adversarial cancellation dot product with a known f64 ground
truth.

It falsified it. On Apple/Metal, df64 and plain f32 accumulation came out
**byte-identical**. Same error, 7.552e0, to the digit.

That isn't a bug in our code. It's the compiler, and we traced it to the
source. Double-double is built on error-free transforms like
`two_prod(a,b): p = a*b; e = fma(a, b, -p)`, where `e` is the exact rounding
tail of the product. Metal's fast-math is on by default and treats float ops as
exact and associative, so it proves `fma(a, b, -a*b) == 0` and folds every
error term to zero at compile time. The double-double collapses back to f32
before it ever runs.

And there's no switch. We read through wgpu-hal's Metal device code, naga's MSL
backend, and the whole public wgpu 30 surface. None of them expose a fast-math,
precise, or math-mode control, and `MTLCompileOptions.fastMathEnabled` defaults
to `YES`.

Two consequences worth stating plainly. First, the *existing* compensated
accumulation (a Neumaier sum in the M0 reduction kernel) has been collapsing to
plain f32 on Metal since day one, for the same reason. That had zero impact on
results, and precisely because the CPU-f64 certificate was the real guarantee
all along. It caught what the shader comment promised and the hardware quietly
didn't deliver. Second, even on an IEEE-strict backend where the transforms
survive, our reduction still narrows each workgroup's double-double partial
back to a single f32 before the cross-workgroup combine. A real precision win
needs df64-typed scratch buffers end to end, not just df64 math inside a
kernel.

So df64 is deferred, with a written source-level reason and a concrete list of
what would have to change to revisit it. We also don't claim the one afiro
solve where f32 and df64 finished with different iteration counts as a df64
win. That's chaotic sensitivity to rounding order in an iterative method, noise
in both directions, not a precision gain.

The honest version of "the double-double trick alone is a good post" turned out
to be this: the post is about why it doesn't work on the most popular WebGPU
backend, and how you can prove that from the compiler down instead of guessing.

## 20 of 32, and the 12 we were wrong about

Sundial runs the classic Netlib LP set through a CLI sweep. One row per
instance, each solved on the GPU and then CPU-f64-verified, reported against
the published Netlib optima. The split on 32 instances:

**20 of 32 reach `Optimal`** at CPU-f64-verified 1e-4. The other **12 stop at
`IterationLimit`**. There are **no parse failures**, down from 1 in M1.

Among the 20 Optimal rows, every instance matches the published optimum to
better than 1e-3. Worst real case is adlittle at 6.7e-4. The apparent worst,
e226 at 3.6e-1, is a footnote rather than an error: its Netlib-readme optimum
uses the opposite sign convention for the objective-row RHS constant, so our
KKT-certified −11.635074 reads as a large relative error against the readme's
≈ −18.75. The report annotates that in a note column rather than quietly
"fixing" the number.

`blend.mps` is the instance that changed this milestone. It used to fail the
parser on a set-name-less RHS line, a real-world MPS corner, and now it parses
and solves to `Optimal` at −30.8119660669 against the optima file's
−30.812149846.

Now the part where we were wrong.

For two milestones we attributed those 12 `IterationLimit` rows to the f32
iterates and called it "the documented f32 wall." That was a misdiagnosis, and
the correction is worth more than the original claim was. Re-solving all 12 on
the CPU **f64** reference reproduces every failure, in double precision, where
there is no 1e-4 floor to hit.

What they actually show is step imbalance. One residual collapses to machine
epsilon while the other stalls orders of magnitude above tolerance: agg runs a
primal of 1.6e-16 against a dual of 1.3e-2, sc205 a primal of 8.4e-17 against a
gap of 0.77. And the stalled side flips between instances, so no single fixed
step ratio can serve all of them. The explicit path runs with τ = σ and no
primal weight, which is exactly the gap.

An opt-in movement-based ω (PDLP's ‖Δy‖/‖Δx‖ rule) takes the same GPU sweep, at
the same 2M cap, from 20/32 to **30 of 32 with no status regressions**, and
cuts total iterations across the already-solving instances by **5.2×**.
beaconfd alone goes from 1,680,896 iterations to 10,432.

It ships **off by default**, and the table above is the default-off run,
because the headline undersells a real cost. Two of the newly-Optimal
instances, lotfi and bnl1, land 5.8e-2 and 1.2e-2 from the published optima.
Both are honest `Optimal`: each passed the independent f64 KKT recheck at its
returned point. But a KKT residual under 1e-4 doesn't tightly bound objective
error on degenerate instances, and those two sit well outside the 1e-3 band
every row in the table occupies. Trading accuracy you can measure for a status
count you can advertise is the exact trade this project refuses to make
silently, so the flag stays opt-in.

**On comparing against published GPU-LP benchmarks.** We looked for overlap
with the Mittelmann benchmarks at plato.asu.edu, the standard reference, which
does include cuOpt and cuPDLPx. There's none to cite honestly. Those benchmarks
target data-center instances: the small end of their GPU feasibility set
(qap15, about 6,331 rows by 22,275 columns) already dwarfs every instance in
our classic-Netlib sweep, the large end runs to tens of millions of rows and
columns, and the GPU solvers there run on an NVIDIA B200. None of our 32
instances appear in them, at different tolerance tiers on different-class
hardware. Rather than manufacture an apples-to-oranges table, we state absolute
numbers and point at <https://plato.asu.edu/bench.html>. The comparison that
*is* honest is the one anyone can run: open the demo, solve a million-variable
problem, time it on your own GPU.

## Where this breaks

A responsible reading needs the limits stated as plainly as the wins.

**f32 iterates, 1e-4 default tier.** The iterate arithmetic is single precision
and the headline tolerance is PDLP's "moderate accuracy" tier. Tighter
tolerances are genuine future work, and per the df64 section the obvious route
is blocked on Apple/Metal until wgpu exposes a strict-math control.

**Not every Netlib instance solves in the default configuration.** The 12
`IterationLimit` rows are honest non-solves. The opt-in ω closes 10 of them,
and we left it opt-in because two of the instances it converts land 1–6% off
the published optima. That's a worse trade than the headline count suggests.

**No presolve.** We evaluated the recent GPU-presolve work (Cederberg & Boyd,
arXiv 2604.23951) and deferred it. It only applies to the explicit-CSR path,
never the matrix-free transport path, its wins concentrate in a minority of
large redundant instances, and the postsolve mapping that keeps certificate
honesty intact is multi-week correctness-critical work.

**Metal fast-math caveat.** Any compensated accumulation is collapsed to plain
f32 on the Metal backend. The CPU-f64 certificate is what makes that safe.
Don't read the shader comments as a precision guarantee.

**Small LPs still belong to simplex on a CPU.** A first-order GPU method pays
off at scale. The million-variable transport hero is where the architecture
earns its keep, not afiro. Simplex beats us on the little ones, and that's
expected.

**WebGPU availability floor.** You need Chrome or Edge 113+, Firefox 141+, or
Safari 26+. Reported `solve_ms` is the solve loop only, excluding host
preprocessing (Ruiz scaling and the power-iteration norm), and that convention
is consistent across the CPU and GPU engines.

None of those change what the demo does on your machine.

## Try it

**In the browser**, with nothing to install:

> <DEMO_URL>

Pick a preset (blobs, ring, spiral, checker, corners) or draw your own source
and target masses with the brush. Choose 16×16 or 32×32. Watch the three live
heatmaps and the convergence chart. There's a drop-a-file benchmark page too:
hand it an `.mps` or `.mps.gz` and get an honest results table back. And a taxi
page that dispatches every open ride in Manhattan from the 2015 TLC record,
greedily first and then to a CPU-verified optimum.

**As a library**, from npm:

```bash
npm install sundial-lp
```

```js
import init, { solveMps } from "sundial-lp";
await init();

// Runs on the browser's GPU, resolves once the result has been
// re-verified in f64 on the CPU.
const result = await solveMps(mpsText, 1e-4, (p) => {
  // p = { iter, rel_primal, rel_dual, rel_gap, ms_per_iter }
});
```

Private data never leaves the browser. The optimization runs client-side on the
user's own GPU, with no backend.

**On the command line**, native Metal, Vulkan, or DX12 through `wgpu`:

```bash
cargo install sundial-cli

sundial solve path/to/model.mps
sundial transport --grid 32                          # 1,048,576 variables
sundial bench bench/netlib --out results.csv
sundial report results.csv --out report.md
```

Source is Rust, `wgpu`, and WGSL, dual-licensed MIT OR Apache-2.0. One codebase
runs on the CPU in f64, on a native GPU, and in a browser tab.

Open the demo, pick 32×32, and watch the primal residual come down. That curve
is the whole argument.
