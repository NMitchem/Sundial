# Sundial

The first linear-programming solver that runs as WebGPU compute shaders —
on any GPU (Apple, AMD, Intel, NVIDIA), natively or in a browser tab, with
zero install and no CUDA anywhere.

**Status: M1.** Restarted PDHG (the PDLP algorithm family) in f32 WGSL,
now with matrix-free operators: a 1,048,576-variable optimal-transport
problem (two 32×32 grids) solves to verified 1e-4 in ~9.4 s, natively and
in a browser tab, with no constraint matrix ever materialized. A
drop-a-file benchmark page and a Netlib sweep round out the CLI tooling.
Every "Optimal" is still re-verified on the CPU in f64 — the GPU never
grades its own homework. The optimality certificate is evaluated at a
sign-projected dual, so the duality gap is genuinely enforced — not
silently dropped — on standard-form problems.

## Try it

    # native CLI (Metal/Vulkan/DX12 via wgpu)
    cargo run -p sundial-cli --release -- solve crates/sundial-mps/tests/fixtures/afiro.mps

    # 1M-variable optimal-transport hero (two 32x32 grids, ~9.4s on Apple M4 Pro/Metal)
    cargo run -p sundial-cli --release -- transport --grid 32

    # browser demo
    wasm-pack build crates/sundial-mps --target web
    cd web && npm install && npm run dev
    #   -> index.html  (transport hero: presets, grid picker, live heatmaps + convergence chart)
    #   -> bench.html  (drop a .mps / .mps.gz file, get an honest results table)

## Tests

    cargo test --workspace                      # CPU suite (CI)
    cargo test --workspace -- --include-ignored # + GPU suite (needs a GPU; includes the 1M gate in --release)

## Honest limits (M1)

- f32 iterate arithmetic; headline tolerance is still the 1e-4 tier (PDLP's
  "moderate accuracy"). Tighter tolerances are future work (double-double,
  M2).
- No presolve, no infeasibility certificates yet (infeasible models hit the
  iteration cap and are reported as such, never as solved).
- Netlib sweep (32 instances): 19/32 reach verified 1e-4 Optimal; the other
  12 stop honestly at IterationLimit — that's the documented f32 wall, not
  a silent failure — and 1 hits a parser limitation (blend.mps's
  set-name-less RHS format, M2 backlog). One footnote: e226's netlib-readme
  optimum uses the opposite sign convention for the objective-row RHS
  constant, so our verified −11.635074 reads as a large relative error
  against the readme's ≈ −18.75 — that's a convention mismatch, not a
  solver defect.
- Simplex on a CPU still wins on small LPs — the GPU pays off at scale,
  which is what the 1M-variable transport hero demonstrates.

## Layout

- `crates/sundial-core` — solver: problem types, scaling, CPU f64 reference
  PDHG, GPU engine + WGSL kernels (matrix-free `LinOp` operators)
- `crates/sundial-mps` — MPS parser (+ wasm bindings for the web demo)
- `crates/sundial-cli` — `sundial solve` / `sundial transport` / `sundial bench` / `sundial report`
- `web/` — Vite + TS demo: transport hero page + benchmark page

License: MIT OR Apache-2.0.
