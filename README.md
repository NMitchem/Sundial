# Sundial

The first linear-programming solver that runs as WebGPU compute shaders —
on any GPU (Apple, AMD, Intel, NVIDIA), natively or in a browser tab, with
zero install and no CUDA anywhere.

**Status: M0.** Restarted PDHG (the PDLP algorithm family) in f32 WGSL,
solving small Netlib instances to relative KKT ≤ 1e-4. Every "Optimal" is
re-verified on the CPU in f64 — the GPU never grades its own homework.
The optimality certificate is evaluated at a sign-projected dual, so the
duality gap is genuinely enforced — not silently dropped — on standard-form
problems.

## Try it

    # native CLI (Metal/Vulkan/DX12 via wgpu)
    cargo run -p sundial-cli --release -- solve crates/sundial-mps/tests/fixtures/afiro.mps

    # browser demo
    wasm-pack build crates/sundial-mps --target web
    cd web && npm install && npm run dev

## Tests

    cargo test --workspace                      # CPU suite (CI)
    cargo test --workspace -- --include-ignored # + GPU suite (needs a GPU)

## Honest limits (M0)

- f32 iterate arithmetic; headline tolerance is the 1e-4 tier (PDLP's
  "moderate accuracy"). Tighter tolerances are future work (double-double).
- No presolve, no infeasibility certificates yet (infeasible models hit the
  iteration cap and are reported as such, never as solved).
- Small instances only so far; simplex on a CPU beats this on small LPs —
  the GPU pays off at scale, which is what M1 (1M-variable optimal
  transport in-browser) is for.

## Layout

- `crates/sundial-core` — solver: problem types, scaling, CPU f64 reference
  PDHG, GPU engine + WGSL kernels
- `crates/sundial-mps` — MPS parser (+ wasm bindings for the web demo)
- `crates/sundial-cli` — `sundial solve` / `sundial bench`
- `web/` — Vite + TS demo page

License: MIT OR Apache-2.0.
