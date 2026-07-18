# Taxi demo — "Every open ride in Manhattan, dispatched perfectly" (design)

**Date:** 2026-07-18 · **Status:** approved design, pre-plan
**Goal:** a zero-explanation, viral-shareable demo page that makes Sundial's browser-GPU solving feel like magic to a CS-curious visitor who neither knows nor cares what an LP is.

**The pitch (the sentence a visitor texts a friend):**
> "This site paired 1,024 taxis with 1,024 riders in Manhattan — the provably best pairing out of a million routes — in seconds, on my GPU, in my browser. Then I tapped the map to add myself and watched the whole city re-plan around me."

## Placement

A third page, `taxi.html`, in the existing `web/` Vite app (alongside `index.html` hero and `bench.html`), consuming the local wasm-pack output like the other pages. Not a separate npm-consumer app. The personal website (`noah-mitchem-os`, Vite+React, separate repo) links to the deployed demo page — a project card there is follow-up work outside this spec.

## Experience (five beats)

1. **Scene.** Dark minimal Manhattan: bundled island-outline/street GeoJSON rendered to canvas — no map tiles, no external requests. Yellow dots = free cabs, white dots = open ride requests. Caption: "Manhattan, Friday 6:03 PM. 1,024 open ride requests, 1,024 free cabs. Positions from real NYC taxi records."
2. **The obvious way.** A single DISPATCH button. Greedy nearest-neighbor dispatch (pure JS, instant) draws red lines into a visible tangle; a counter totals pickup miles.
3. **The perfect way.** "Now the best answer that exists." GPU solve runs with live telemetry (real iterations via the existing progress callback, ~5–10 s at proven hero scale). On verified completion, every red line morphs (~800 ms) from its greedy cab to its optimal cab: a calm green web with zero crossings. Counter rolls down; the greedy-vs-optimal gap is displayed as measured, never scripted. Then the dare: "523,776 pairs of lines. Find two that cross. You won't."
4. **The poke.** "Tap anywhere. You need a cab." Tap appends a rider, the city re-solves from scratch (same telemetry moment), the visitor's cab lights up, and a ripple flashes every changed pairing: "You just changed N other people's cabs."
5. **Receipts** (footer): possible-routes count, iterations, solve time, adapter name, "optimality independently verified (CPU f64)", link to the Sundial repo/writeup.

## Scale decision (approved)

1,024 × 1,024 — exactly the proven browser hero scale (~1.05 M variables), not 10,000². Copy frames it honestly as *this minute's open requests* (realistic for rush hour; the data really is a TLC slice). A 10k² dense instance (100 M variables) would blow solve pacing and buffer budgets; a sparse k-nearest variant is explicitly out of scope (see Out of scope).

## Architecture

### Data (offline, one-time)

- New script in `scripts/` (reproducible, in the spirit of `fetch_netlib.sh`) pulling one rush-hour window from public NYC TLC yellow-cab trip records: pickups → rider positions; drop-offs from the preceding minutes → free-cab positions (a stated proxy, disclosed in the page footer).
- Output: one static JSON (~40 KB gz) of 1,024 + 1,024 points checked into `web/public/`, plus a bundled Manhattan outline GeoJSON. Raw downloads stay gitignored.

### Core additions (Rust, small; engine/shaders/certificates untouched)

1. `transport::problem_from_points(src: &[Point], tgt: &[Point]) -> OpProblem<TransportOp>` — mirrors `problem_from_masses`, but `c` is pairwise Euclidean distance between real coordinates. `TransportOp` already supports unequal `ns`/`nt` and an arbitrary explicit cost vector; no operator or WGSL changes.
2. Wasm entry `solveMatching(srcXY, tgtXY, tol, onProgress)` in `sundial-web`: returns the existing verified stats **plus** the assignment as a `Uint32Array` (dominant plan entry per rider, extracted in Rust — 1,024 ints, not the 4 MB plan). Extraction asserts per-row dominance; with generic real-coordinate costs the optimum is a permutation, and the page copy says "routes of the certified-optimal plan" rather than overclaiming per-line exactness at 1e-4 tolerance.

### Frontend (`web/src/taxi/`, TypeScript, no new dependencies)

`data.ts` (load points) · `map.ts` (canvas basemap + dot layers) · `greedy.ts` (nearest-neighbor dispatch + cost total; pure function) · `lines.ts` (line layer; draw-in + greedy→optimal endpoint morph) · `ripple.ts` (assignment diff, changed-pairing flash, count) · `telemetry.ts` (progress panel on the existing `onProgress` events) · `main-taxi.ts` + `taxi.html` (shell, narrative copy).

### Data flow

Static JSON → JS coordinate arrays → `solveMatching` → GPU solve with progress events → CPU-verified stats + assignment array → greedy layer morphs to optimal → each tap appends a rider and repeats the same path from scratch.

## Error handling

- **No WebGPU:** `webgpuAvailable()` gates the page; fallback shows a pre-recorded video loop of the snap (also the shareable asset) plus a "works live in Chrome/Edge/Safari" note.
- **Uncertified solve:** if a run returns anything but verified `Optimal`, show the best-found dispatch with an honest banner and strip all "proven/perfect" copy for that run. No displayed status the CPU verifier didn't grant.
- **GPU init failure / device lost:** error card with retry. Taps during an in-flight solve are queued, not dropped.

## Testing

- **Rust unit:** `problem_from_points` on n ≤ 6 instances vs. brute-force enumeration over all permutations; dominant-entry extraction recovers the permutation on generic costs, plus a tie-cost case pinning assert behavior; CPU `reference::solve_op` parity on small point clouds. Existing `testgen`/KKT idiom.
- **GPU (`#[ignore]`, local):** one integration test solving the real bundled 1,024×1,024 JSON fixture → asserts verified `Optimal` and assignment-is-a-permutation; logs timing. This is the demo's regression gate.
- **Web:** existing gates only (`tsc --noEmit`, `vite build`, wasm clippy) + the manual browser human gate, as in M2. No new JS test infra.
- CI stays green (GPU tests remain ignored). No existing invariant is weakened.

## Honesty rules (carried into copy)

Data provenance and the free-cab proxy disclosed on-page; greedy comparison measured live; "proven/best possible" language appears only on CPU-verified `Optimal` runs; receipts footer always shows the real status, iterations, and timing.

## Out of scope (recorded, not planned)

- Warm-started re-solves for snappier pokes (natural PDHG extension; stretch only).
- Sparse k-nearest 10,000×10,000 variant via the explicit CSR path.
- NBA schedule lower-bound demo (the moonshot sequel).
- Personal-site project card linking to the deployed page.
