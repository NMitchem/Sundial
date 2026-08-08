# Sundial

The first linear-programming solver that runs as WebGPU compute shaders —
on any GPU (Apple, AMD, Intel, NVIDIA), natively or in a browser tab, with
zero install and no CUDA anywhere.

**Status: M2 — launch-ready, unpublished.** Restarted PDHG (the PDLP algorithm
family) in f32 WGSL, with matrix-free operators: a 1,048,576-variable
optimal-transport problem (two 32×32 grids) solves to verified 1e-4 in ~9.4 s
natively (Apple M4 Pro / Metal), with no constraint matrix ever materialized.
The same wasm/WebGPU code solves it in a browser tab — confirmed
interactively: Optimal (CPU-f64-verified), 16,000 iterations at ~0.55
ms/iter. Every "Optimal" is still re-verified on the CPU in f64 — the GPU
never grades its own homework — and the certificate is evaluated at a
sign-projected dual, so the duality gap is genuinely enforced, not silently
dropped, on standard-form problems. M2 extends the same discipline to the
other two things an LP can be: **Infeasible** and **Unbounded** are only ever
set after a CPU-f64 Farkas certificate (dual ray / primal recession ray)
verifies the GPU's streak-detector nomination. On Netlib's real
infeasible/unbounded set, 2 of 6 instances certify (the other 4 stop at an
honest `IterationLimit`) and zero produce a false `Optimal` — that's the
honest recall number, not a tuning shortfall. The browser demo now has 5
transport presets (blobs, ring, spiral, checker, corners) plus draw-your-own
source/target masses, and the solver ships as an npm package, `sundial-lp`
(inspected via `npm pack`, not yet published).

## Try it

    # smallest possible starting point: an LP you can check by hand
    cargo run --example tiny_lp

    # native CLI (Metal/Vulkan/DX12 via wgpu)
    cargo run -p sundial-cli --release -- solve crates/sundial-mps/tests/fixtures/afiro.mps

    # 1M-variable optimal-transport hero (two 32x32 grids, ~9.4s on Apple M4 Pro/Metal)
    cargo run -p sundial-cli --release -- transport --grid 32

    # browser demo
    wasm-pack build crates/sundial-web --target web   # or: bash scripts/build_npm.sh
    cd web && npm install && npm run dev
    #   -> index.html  (transport hero: 5 presets + draw-your-own, grid picker, live heatmaps + convergence chart)
    #   -> bench.html  (drop a .mps / .mps.gz file, get an honest results table)

- `taxi.html` — every open ride in Manhattan (real 2015 TLC data), dispatched
  greedily and then to a CPU-verified optimum on your GPU; tap to add yourself
  and watch the city re-plan.

## As a library (npm)

    npm install sundial-lp   # not yet published — see RELEASE.md

```js
import init, { solveMps, webgpuAvailable } from "sundial-lp";

await init();
if (!webgpuAvailable()) throw new Error("WebGPU required (Chrome/Edge 113+, Firefox 141+, Safari 26+)");
const result = await solveMps(mpsText, 1e-4, (p) => console.log(p.iter, p.rel_gap));
console.log(result.status, result.objective); // "Optimal (CPU f64 verified)", …
```

## Tests

    cargo test --workspace                      # CPU suite (CI)
    cargo test --workspace -- --include-ignored # + GPU suite (needs a GPU; includes the 1M gate in --release)

## Honest limits (M2)

- f32 iterate arithmetic; headline tolerance is still the 1e-4 tier (PDLP's
  "moderate accuracy"). A double-double (df64) accumulator was attempted and
  is **deferred** — not future work, a closed experiment: on Metal, forced
  fast-math folds the error-free transforms df64 depends on back to plain
  f32 at compile time, so df64 is provably byte-identical to f32 on this
  backend (`docs/notes/df64-experiment.md`).
- No presolve. GPU-presolve (Cederberg & Boyd, arXiv 2604.23951) was
  evaluated and deferred: it only applies to the explicit-CSR path (never
  the matrix-free transport path), and correctness-critical postsolve
  engineering is multi-week (`docs/notes/gpu-presolve-memo.md`).
- Infeasible/Unbounded detection is deliberately conservative: on Netlib's
  real infeasible/unbounded set, 2 of 6 instances certify; the other 4 stop
  honestly at `IterationLimit`. Zero false `Optimal` claims — a missed
  detection costs iterations, a false claim would cost trust.
- Netlib sweep (32 instances): 20/32 reach verified 1e-4 Optimal; the other 12
  stop honestly at `IterationLimit`, not a silent failure; and 0 parse errors
  (blend.mps's set-name-less RHS format, an M1-era parser gap, is now handled).
  One footnote: e226's netlib-readme optimum uses the opposite sign convention
  for the objective-row RHS constant, so our verified −11.635074 reads as a
  large relative error against the readme's ≈ −18.75 — that's a convention
  mismatch, not a solver defect.
- **Correction (2026-08-08):** those 12 rows were described here as "the f32
  wall." That was wrong, and we tested it rather than leave it standing. The
  CPU **f64** reference fails the same 12 the same way — with one side at
  machine epsilon (~1e-16) while the other stalls above tolerance — so
  precision is not the limit. The cause is primal-weight imbalance on the
  explicit path, which runs unweighted. An opt-in movement-based ω
  (`--movement-weight`) takes the same GPU sweep to **30/32 with no status
  regressions**, and cuts iterations 5.2× across the instances that already
  solved. It is **off by default** and the table above is the default-off run,
  because the win is not free: two newly-Optimal instances (lotfi, bnl1) land
  1–6% off the published optima — honest `Optimal` by the KKT certificate, but
  outside the ≤1e-3 accuracy band the rows above sit in. Full numbers and the
  open default-flip decision: `docs/STATUS.md` (M3).
- Simplex on a CPU still wins on small LPs — the GPU pays off at scale,
  which is what the 1M-variable transport hero demonstrates.

## Layout

- `crates/sundial-core` — solver: problem types, scaling, CPU f64 reference
  PDHG, Farkas certificate verification, GPU engine + WGSL kernels
  (matrix-free `LinOp` operators)
- `crates/sundial-mps` — MPS parser (pure parser, no wasm bindings)
- `crates/sundial-web` — wasm bindings for the solver, published as the
  `sundial-lp` npm package
- `crates/sundial-cli` — `sundial solve` / `sundial transport` / `sundial bench` / `sundial report`
- `web/` — Vite + TS demo: transport hero page + benchmark page

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) — it covers the build, the CI gates, and
the invariants (certificate honesty above all) that PRs are held to. Open an issue
before large changes; several obvious-looking improvements are already closed
experiments documented in `docs/notes/`.

Security reports go through
[private vulnerability reporting](https://github.com/NMitchem/Sundial/security/advisories/new),
not public issues — see [`SECURITY.md`](SECURITY.md).

## License

MIT OR Apache-2.0, at your option — see [`LICENSE-MIT`](LICENSE-MIT) and
[`LICENSE-APACHE`](LICENSE-APACHE).

Third-party data bundled here (the Netlib LP fixtures and the NYC TLC taxi
extract) carries its own provenance — see [`NOTICE`](NOTICE).
