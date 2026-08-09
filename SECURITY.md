# Security Policy

## Supported versions

Sundial is pre-1.0. Security fixes go to the latest release on `main` and the
latest published `sundial-lp` npm package. Nothing older is supported.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub's private vulnerability reporting:
[Security → Report a vulnerability](https://github.com/NMitchem/Sundial/security/advisories/new).

Expect an initial response within 7 days. If a report is confirmed, the fix and
the advisory are published together.

## Scope

Sundial is a solver library and a static browser demo. It has no server, no
account system, and stores no user data. The realistic attack surface is:

- **Untrusted MPS input.** `crates/sundial-mps` parses attacker-controlled text,
  and the `bench.html` demo page parses whatever file a user drops in. Panics,
  unbounded allocation, or hangs reachable from malformed MPS are in scope.
- **The `sundial-lp` wasm package.** Memory-safety or sandbox-escape issues in
  the wasm bindings (`crates/sundial-web`) are in scope.
- **Supply chain.** Problems with the published npm tarball's contents are in
  scope: unexpected files, or wrong integrity metadata.

Out of scope:

- **Wrong answers are bugs, not vulnerabilities.** An instance that returns
  `IterationLimit` instead of `Optimal`, or a slow convergence case, belongs in a
  normal issue. See the "What it can't do yet" section of `README.md` for the
  known accuracy ceiling.
- Denial of service from *legitimately* large problems. The solver is expected
  to be slow on hard instances.
- Issues in GPU drivers, browsers, or wgpu itself. Report those upstream, though
  we appreciate a heads-up if Sundial is a practical trigger.

## A note on the certificate invariant

Sundial's core safety property is that a status claim is never made on GPU
evidence alone: `Optimal` requires an independent CPU-f64 KKT recheck, and
`Infeasible`/`Unbounded` require a CPU-f64 Farkas certificate. **A reproducible
case where Sundial reports a verified status that is actually false is the most
serious class of bug in this project.** Report it privately via the link above,
even though it is a correctness issue rather than a conventional vulnerability.
