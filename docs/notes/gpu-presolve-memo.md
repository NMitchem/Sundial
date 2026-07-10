# GPU-presolve memo (arXiv 2604.23951) — M3 decision input

## What the paper does

Cederberg & Boyd (Stanford, April 2026, arXiv 2604.23951) argue presolve for
GPU-accelerated first-order LP solvers should be lightweight, since presolve
stays CPU-bound and sequential while the GPU core step keeps getting faster,
growing presolve's share of total runtime over time. They frame presolve as
atomic "reductions" (fixing a variable, removing a constraint, adding a
multiple of one equality constraint to another, substituting a variable from
a single equality constraint, changing bounds) found by primal/dual
"explorers," taxonomized by cost into fast (singleton rows/columns, doubleton
rows, redundant constraints, variable locks — computed from internal
statistics), medium (parallel row/column detection, primal/dual bound
propagation), and slow (linear-dependence via rank-revealing factorization,
variable substitution, symmetry, sparsification, dominated columns) tiers.
Their open-source implementation, PSLP, uses only the fast+medium tiers,
omitting slow explorers entirely. Tested against Gurobi's presolver and
against cuPDLPx (a GPU PDHG solver) on two public LP benchmark collections,
PSLP captures ~90-94% of Gurobi's nonzero-reduction while running
~6.6-11.8x faster, and enabling it improves cuPDLPx's end-to-end solve time
in aggregate — though the win rate is only ~52-59% of instances, since for
problems cuPDLPx already solves quickly, presolve overhead is net-negative,
with the largest gains concentrated in a minority of large, previously-slow
instances.

## Applicability to Sundial

Precision/WGSL-capability concerns are moot: this presolve is pure CPU
preprocessing that never touches the GPU — the authors explicitly argue
presolve belongs on the CPU and dismiss a prior GPU-accelerated
primal-propagation paper as unrealistic. So none of Sundial's WGSL
portable-subset limits (no subgroups, no f16/f64, Metal fast-math — see
docs/notes/df64-experiment.md) are engaged by anything here; it would run
entirely in host-side Rust before any WGSL kernel executes. The paper never
discusses numerical precision requirements for the presolve arithmetic
itself. The real incompatibility is the matrix-free path: PSLP stores the
constraint matrix in both CSR and CSC form, with reductions rewriting it in
place (e.g., fill-in from combining rows), and the paper never discusses an
operator-based / matrix-free alternative. This technique can only ever apply
to Sundial's explicit-CSR path, never the matrix-free transport path.
Postsolve reconstructs a full original-problem primal-dual solution
satisfying the original KKT/complementary-slackness conditions for optimal
solutions; I found no discussion of postsolve mapping for
infeasibility/unboundedness (Farkas) certificates — unclear from the paper.

## Cost estimate

A minimal port of the fast-tier explorers alone (singleton rows/columns,
doubleton rows, redundant constraints, variable locks) is the natural
minimal useful subset — the paper's own tier split calls these cheapest, and
PSLP-as-shipped (fast+medium) already gets ~90-94% of Gurobi's reduction, so
fast-only would capture a smaller but still meaningful slice. The real cost
is in postsolve: every reduction needs a correctness-critical inverse mapping
back to the original problem before Sundial's CPU-f64 KKT re-verification
runs, and a bug there breaks certificate honesty silently rather than
failing loudly. That is a multi-week Rust implementation and test-parity
effort, scoped to the explicit-CSR path only, sitting before Ruiz+PC scaling.
Wrapping PSLP directly via FFI is a cheaper alternative but adds a C
dependency and still needs Sundial-side postsolve glue.

## Recommendation

DEFER — the technique only applies to the explicit-CSR path (not
matrix-free), the paper's own data shows wins concentrated in a minority of
large/slow instances with real overhead risk elsewhere (win rate ~52-59%),
and postsolve-correctness engineering for certificate honesty is
non-trivial; revisit at M3+ once explicit-CSR-path benchmarking shows
Sundial has large/redundant instances where presolve-shaped headroom
actually exists.
