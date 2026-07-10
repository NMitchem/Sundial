# GPU-presolve memo (arXiv 2604.23951) — M3 decision input

Paper: Cederberg & Boyd, "Presolving for GPU-Accelerated First-Order LP
Solvers," Stanford, April 2026. Read in full (main text, pp. 1-20; pp. 21-24
are references only).

## What the paper does

The paper argues presolve for GPU first-order LP solvers should be
lightweight because presolve stays CPU-bound and sequential while the
GPU-accelerated core step gets faster, so presolve's *share* of total runtime
grows over time. It frames presolve as a sequence of atomic *reductions*
(fixing a variable, removing a constraint, adding a multiple of one equality
constraint to another, substituting a variable from a single equality
constraint, changing variable bounds), each found by a *primal* or *dual*
*explorer* and each paired with a postsolve transform. Explorers are
taxonomized by cost: **fast** (singleton rows, doubleton rows, redundant
constraints, column singletons in an equality/inequality, variable locks —
computed from internal statistics, no extra passes), **medium** (parallel
row/column detection, primal/dual bound propagation — needs hashing or
pairwise comparison), and **slow** (linear-dependence detection via rank-
revealing factorization, variable substitution with fill-in control, variable
symmetry, primal/dual sparsification, dominated columns). Their open-source
implementation, PSLP (C, single-threaded with 2-4 POSIX threads for some
parts; github.com/dance858/PSLP; now integrated into cuPDLPx, NVIDIA's cuOpt,
and HPR-LP), implements only the fast+medium tiers and omits slow explorers
entirely.

Tested on Mittelmann's LP collection (48 instances, >10^5 nonzeros) and 383
MIPLIB 2017 root-node LP relaxations, split into small/medium/large by nnz
count (10^5-10^6 / 10^6-10^7 / >10^7). Presolve ran on a CPU (M4 Pro, 14
cores); the reduced problem solved via cuPDLPx on an H100. Reported numbers:
PSLP captures ~90% (Mittelmann) / ~94% (MIPLIB) of Gurobi's nnz reduction
while being ~11.8x / ~6.6x faster on average per-instance; in aggregate
(Table 2) PSLP's mean presolve time is 0.33s AM / 0.084s GM (Mittelmann) and
0.057s AM / 0.013s GM (MIPLIB), versus Gurobi's 3.59s/0.70s and 0.47s/0.077s.
End-to-end, enabling PSLP improves cuPDLPx's shifted geometric mean solve
time in every size bucket on both datasets, most dramatically on large
problems (Mittelmann large SGM10: 169.33s -> 63.05s; MIPLIB large SGM10:
408.26s -> 44.12s — the paper's headline ">2.5x" and ">9x" claims). But the
win is not uniform: overall win rate is only ~52% (Mittelmann) / ~59%
(MIPLIB) — for problems cuPDLPx already solves fast, presolve overhead is
net-negative, and the large gains concentrate in a minority of previously-slow
instances (Figure 3 shows orders-of-magnitude speedup only at the tail).
Presolve's own overhead (Table 4) is 11.8%/12.1% of reduced-problem solve
time on average for PSLP (up to 19.3%/39.1% for large problems), versus
76.9%/56.0% for Gurobi (up to 82.3%/198.3% for large — Gurobi's presolve can
take ~2x longer than solving the reduced problem).

## Applicability to Sundial

Precision/WGSL-capability question is moot: presolve in this paper is pure
CPU preprocessing and never touches the GPU. The authors explicitly argue
presolve belongs on the CPU (sequential, branch-heavy, irregular-memory-
access transformations) and dismiss a prior GPU-accelerated primal-
propagation paper (SGP22) as unrealistic (it needs up to 100 propagation
rounds versus the "typically only a few rounds" PSLP/Gurobi use in practice).
So none of Sundial's WGSL portable-subset constraints (no subgroups, no
f16/f64, Metal fast-math — see [[df64-experiment]] /
`docs/notes/df64-experiment.md`) are engaged by anything in this paper; it
would live entirely in host-side Rust before any WGSL kernel runs. The paper
does not discuss numerical precision (f32 vs. f64) requirements for the
presolve arithmetic at all — unclear from the paper, not inferred either way.

The matrix-free path is where the real incompatibility is, and it is total,
not partial. PSLP stores the constraint matrix in *both* CSR and CSC form
(§3.2), with per-row/column slack to absorb fill-in from reductions that
combine rows (e.g. adding a multiple of one equality constraint to another),
and must keep the two representations in sync as reductions mutate the
matrix in place. There is no discussion anywhere in the paper of an operator-
based or matrix-free representation — presolve as described here *requires*
an explicit, mutable sparse matrix to read and rewrite. It therefore can only
ever apply to Sundial's explicit-CSR (Ruiz+PC-scaled) path; it has no
meaning for the matrix-free transport operator path, which has no matrix for
any reduction to act on.

On certificate honesty: the paper's reduction abstraction pairs every
presolve transform with a postsolve transform that reconstructs a full
primal-dual solution (x*, y*, z*) satisfying the original problem's KKT/
complementary-slackness conditions (their eq. 2). Primal-exploration
reductions preserve the feasible set exactly; dual exploration is split into
*weak* (provably recovers every optimal solution of the original problem)
and *strong* (only guarantees recovery of at least one optimal solution, not
all). This is compatible in principle with re-verifying an *optimal*
certificate against the original problem via postsolve. However, the paper's
postsolve discussion is framed entirely around recovering an optimal
primal-dual solution — I found no explicit treatment of postsolve mapping
for infeasibility or unboundedness certificates (e.g., Farkas rays) through
the presolve/postsolve round trip. Whether Sundial's Farkas-certificate path
would survive presolve is unclear from the paper.

## Cost estimate

A from-scratch minimal port (fast-tier explorers only: singleton rows,
doubleton rows, redundant constraints, column singletons, variable locks) is
the natural "minimal useful subset" — the paper's own tier split already
separates these as needing "minimal computation... scanning internal
statistics," and PSLP-as-shipped (fast+medium) already gets ~90%/94% of
Gurobi's reduction, so fast-only would capture a still-meaningful but smaller
slice at lower cost than PSLP itself. Real engineering cost concentrates in
the postsolve half: every reduction needs a correctness-critical inverse
mapping back to the original problem's indices/duals before Sundial's
CPU-f64 KKT re-verification runs, and a bug there silently breaks certificate
honesty rather than failing loudly. This is a multi-week Rust implementation
and test-parity effort at minimum, scoped to the explicit-CSR path only — it
would need to sit before Ruiz+PC scaling in that pipeline and would not touch
the matrix-free path at all. Alternatively, PSLP could be wrapped via FFI
(it's designed as a solver-independent C library with a minimal
pointer-based interface) to skip the reimplementation, at the cost of adding
a C dependency and still requiring Sundial-side postsolve/certificate-mapping
glue to feed the CPU-f64 verifier.

## Recommendation

DEFER — the technique only applies to the explicit-CSR path (not
matrix-free), the paper's own data shows wins concentrated in a minority of
large/slow instances with real overhead risk elsewhere (win rate ~52-59%),
and postsolve-correctness engineering for certificate honesty is
non-trivial; revisit at M3+ once explicit-CSR-path benchmarking shows
Sundial has large/redundant instances where presolve-shaped headroom
actually exists.
