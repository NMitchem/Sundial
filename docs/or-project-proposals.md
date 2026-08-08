# OR Project Proposals — Survey Results

**Date:** 2026-07-06 · **Method:** 5 parallel web-research scouts (ML×OR frontier, practitioner pain, HN archaeology, tooling gaps, benchmark records) produced 31 raw candidates → checkpoint scoring killed 23 → 8 survivors each got a dedicated adversarial verifier instructed to *kill* its candidate by finding prior art and fetching every load-bearing URL. Result: 3 ALIVE, 2 WOUNDED-but-reshaped (included below in reshaped form), 3 WOUNDED-and-cut. All links below were fetched live during verification (2026-07-06/07) unless noted.

**HN ground truth used for scoring** (from a 35-post >100-point corpus pulled via the Algolia API): what lands is watch-it-happen visual demos (484, 253, 232), records with relatable data (435 for a TSP walking tour of Korean bars), Rust/verified/GPU solver releases (315, 402, 221), and breakthrough explainers (498, 427). What flops: naming the technique or vendor in the title ("vehicle routing" tops out at 71 points ever, "Gurobi" at 14, "branch and bound" at 6) and B2B logistics framing (Onfleet launch: 8 points). Every pitch below is written to hide the jargon inside the story.

---

## Proposal 1 — "Sundial": a GPU optimization solver that runs in any browser tab

### The pitch
The world's biggest planning problems — crew schedules, ad budgets, shipping plans — are solved by a class of math engine that just went through a GPU revolution, but every GPU version (Google's, NVIDIA's, and as of April 2026 even the leading open-source solver's) requires NVIDIA data-center hardware and CUDA. I'll build the first one that runs on *any* GPU — Apple, AMD, Intel, NVIDIA — inside a browser tab, nothing to install. Open a web page, drop in a problem with half a million variables, watch it converge on your own laptop's graphics card.

### The HN title test
**"Show HN: I solved a 500,000-variable optimization problem in a browser tab, on any GPU"**
(alt: "Show HN: A GPU optimization solver in WebGPU — no CUDA, no install, no server")

### Novelty evidence
Verifier ran the kill-searches directly: "WebGPU linear programming", "WebGPU simplex", "WebGPU PDHG", "wgpu linear programming", "WGSL optimization solver", GitHub topic crosses of `webgpu` × {`simplex`, `linear-programming`, `interior-point`}, plus arXiv/HN sweeps — **zero hits**. Closest prior art:

- [cuPDLP-C](https://github.com/COPT-Public/cuPDLP-C) — the reference GPU first-order LP solver. CUDA-only. Crucially, it ships an official single-precision build flag (`SFLOAT`), proving f32 is a supported mode, not a research risk.
- [HiGHS "HiPDLP"](https://highs.dev/assets/HiGHS_Newsletter_26_0.pdf) — announced April 2026; the leading open solver's GPU path explicitly "runs on an NVIDIA GPU," Linux/Windows only. The lock-in is *deepening*, not resolving.
- [MPAX](https://github.com/MIT-Lu-Lab/MPAX) — JAX-based, vendor-portable, but a Python/XLA install, not browser/zero-install.
- [clp-wasm](https://github.com/centrifuge/clp-wasm), highs-js, YALPS — browser solvers exist but are all CPU WASM ports of simplex-era code; Google's own OR-Tools WASM port documents that its GPU-dependent code doesn't survive the browser port.

**Daylight in one sentence:** nobody occupies the intersection of GPU-parallel LP × any GPU vendor × zero-install browser execution — the entire GPU-LP wave is CUDA-locked and the entire browser-solver world is CPU-locked.

### Build plan
**Stack:** Rust + wgpu, WGSL compute shaders (sparse matrix-vector kernels, parallel reductions), restarted PDHG with diagonal preconditioning and adaptive restarts (published recipes from the cuPDLP papers), double-double f32 emulation for accumulation steps only; thin TypeScript shell; also ships as an npm/WASM library, not just a demo page.

- **Milestone 0 (one weekend):** MPS-file parser + PDHG inner loop in WGSL; solve small classic instances (afiro → adlittle) to 1e-4 tolerance with a live convergence chart, racing a CPU WASM solver (glpk-wasm/highs-js) side-by-side on the same page.
- **Milestone 1 (benchmarkable):** run the Netlib set and a subset of Mittelmann's LP benchmark instances; publish a table of solve time + achieved tolerance vs. published CPU (HiGHS, OR-Tools/PDLP) and CUDA (cuOpt, cuPDLPx) numbers — the honest claim is "within X× of data-center CUDA on a consumer Mac, in a browser," which is checkable by anyone who opens the page.
- **Milestone 2 (shareable artifact):** polished site where you drop any MPS/LP file and watch it solve + `npm install` embeddable solver + writeup on f32 first-order methods in WebGPU (the double-double trick alone is a good post).

**Data:** Netlib LP set — http://www.netlib.org/lp/data/ (verified live; ~90 MPS instances). Mittelmann benchmark — https://plato.asu.edu/bench.html (verified live, last updated 2026-07-01; includes cuOpt/cuPDLPx GPU baselines to compare against).

### Kill risks
1. **The f32 numerics wall:** PDHG in single precision may stall before 1e-4 on ill-conditioned instances; mitigations (double-double accumulators, restarts, preconditioning) are known but this is where the 8 weeks actually go — if it only solves easy instances, the demo deflates.
2. **WebGPU ceilings:** storage-buffer size limits and Safari/driver quirks could cap "500k variables" to something less impressive on some machines; the headline number must be validated on mid-range hardware early.
3. **"Who needs this?" skepticism:** client-side LP is a new category, not a replacement for Gurobi; if the npm-library story (private data never leaves the browser, optimization inside web apps with zero backend) doesn't land, it reads as a stunt.

### Rubric scores
- **Legibility: 4** — "giant resource-allocation math on any GPU in a browser tab" survives one paragraph, but you do spend one sentence explaining what the problem class is.
- **Novelty: 5** — two independent search sweeps (Scout A, then the verifier) found literally zero WebGPU implementations of any LP algorithm, and the CUDA lock-in deepened three months ago.
- **Demoability: 4** — weekend demo is a live convergence race, which is compelling but a chart rather than a map.
- **Benchmark truth: 5** — Netlib + Mittelmann (updated five days before this survey) give standard instances with published GPU-solver baselines; every claim is reproducible in the reader's own browser.
- **Right-sized: 4** — the algorithm is published closed-form recipes over sparse matvec + reductions (no factorizations), f32 mode is precedented, but numerics tuning is real work.
- **HN resonance: 5** — anti-CUDA-lock-in + Rust/WASM/GPU + "runs on hardware you already own" hits three proven patterns at once (CreuSAT 315, PDLP 221 as a dry corporate blog, OptaPlanner 402).

**Total: 27/30**

---

## Proposal 2 — "Packbench": machine-verify 30 years of packing records, then attack the neglected shelves

### The pitch
For decades, hobbyists have maintained world-record tables for puzzles like "fit 26 identical circles into the smallest possible square" — as hand-edited web pages with no automatic checking. In 2025, DeepMind and Sakana AI made headlines by beating one of those records with LLM-driven search, proving the records are soft — but they cherry-picked five instances and moved on. I'll build the first machine-readable, automatically-verified registry of packing records across all the shape families, re-certify every published pack, and aim an open LLM-evolution pipeline at the shelves nobody is defending.

### The HN title test
**"Show HN: I built an auto-verifier for 30 years of packing records, then set an LLM loose on the shapes nobody's updated in years"**
(and if the auditor catches errors in published records — a real possibility — that becomes its own post)

### Novelty evidence
- [AlphaEvolve results repo](https://github.com/google-deepmind/alphaevolve_results) (DeepMind, 2025) — verifier confirmed it touched **exactly five** packing instances: circles-in-square n=26 and n=32, circles-in-rectangle n=21, hexagon-in-hexagon n=11/12. Everything else is untouched by the AI labs.
- [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) — open AlphaEvolve clone whose flagship demo is n=26 circle packing; its issue #156 shows a user beating AlphaEvolve's n=26 result just by tuning configs. That single instance is scorched earth — this project deliberately avoids it.
- [ShinkaEvolve](https://github.com/SakanaAI/ShinkaEvolve) (Sakana, Sep 2025) — beat AlphaEvolve's n=26 in ~150 evaluations. Same handful of instances.
- arXiv:2601.05943 — matches AlphaEvolve's packing results **with zero LLMs** (plain nonlinear solvers), which deflates "AI discovers math" theater; this project's framing is accordingly honest (LLM evolves construction/restart strategies; classical optimization does the heavy lifting).
- The record tables themselves: [Packomania](http://www.packomania.com) (verified live, maintained), [Erich Friedman's Packing Center](https://erich-friedman.github.io/packing/) (verified live — pages updated 6/3/26 and 7/3/26), [squares-in-squares continuation](https://kingbird.myphotos.cc/packing/squares_in_squares.html) (verified live, records through Feb 2026). All are hand-maintained HTML; **no machine-readable, auto-verified, cross-family registry exists anywhere** — the verifier searched for packing-record registries and interval-arithmetic verification services and found only decades-old specialist papers (Markót's computer-assisted optimality proofs), never infrastructure.

**Daylight in one sentence:** the labs proved the records are attackable but touched 5 of several hundred table entries and built zero public infrastructure; the registry + feasibility-certifier doesn't exist at all, and the low-attention shape families (circles in pentagons/heptagons, squares in triangles…) have no automated defender.

### Build plan
**Stack:** Rust core — exact/interval arithmetic feasibility certifier (e.g. `inari`/rational arithmetic; certifies non-overlap + containment to rigorous tolerances, explicitly *not* claiming global optimality), scrapers/parsers for the three record sites, SQLite + static-site registry with canvas renderings of every pack; LLM-evolution loop (his existing PDPTW-evolution machinery, retargeted) driving basin-hopping/SA local optimization on GPU/laptop.

- **Milestone 0 (one weekend):** parse Packomania coordinate files for 2–3 families; render packs on canvas; certify feasibility of ~100 published records with interval arithmetic; publish the first "verified ✓" table anyone has ever generated for these pages.
- **Milestone 1 (benchmarkable):** full registry across circles-in-{square, circle, triangle, pentagon…} + squares-in-{square, circle} with per-record certificates and reproducible re-optimization scripts; the benchmark is the published record values themselves — every claim checkable to the digit.
- **Milestone 2 (shareable artifact):** the live registry site + evolution dashboard attacking 2–3 neglected families, with every attempt logged; any genuine improvement gets submitted to Specht/Friedman/Ellsworth for canonical credit (they publish contributor names — that's the record mechanism).

**Data:** the three record sites above (all verified live) — coordinates are published per-record; no other dataset needed.

### Kill risks
1. **Zero new records:** the legible families are defended by active hobbyists with tuned annealers (new records as recent as Dec 2025–Feb 2026); if 8 weeks yields no improvement, the artifact is "just" the registry + auditor — still shippable, but the headline softens from record to infrastructure.
2. **Verification-rigor challenge:** specialists (the Markót lineage) do interval-arithmetic *optimality proofs*; if the project's feasibility-only certificates are oversold as more than they are, the exact community it serves will dunk on it — claims must stay scoped.
3. **Deflated AI narrative:** since plain solvers matched AlphaEvolve, "LLM finds records" framing invites skepticism; the honest "LLM evolves strategies, math does the work" story is stronger but less flashy.

### Rubric scores
- **Legibility: 5** — a single picture of circles in a square explains the entire project.
- **Novelty: 4** — the registry/certifier genuinely doesn't exist (verified), but the attack playbook itself is now well-precedented, so the novel piece is infrastructure + coverage rather than method.
- **Demoability: 5** — canvas pack gallery + live annealing animation is a weekend deliverable and endlessly GIF-able.
- **Benchmark truth: 5** — decades of published records with exact coordinates make every claim checkable to the last decimal, and certificates make it machine-checkable.
- **Right-sized: 4** — laptop-scale nonlinear restarts handle n=20–60; the LLM layer is machinery he already owns; rigorous optimality proofs are correctly out of scope.
- **HN resonance: 5** — records + "hobbyists vs. AI labs" + visual gallery; the sphere-packing record post hit 427 and the Korea-tour record hit 435.

**Total: 28/30**

---

## Proposal 3 — "Swarmroute": watch your GPU test 200,000 route tweaks per frame, in the browser

### The pitch
Improving a delivery route means testing millions of tiny tweaks — swap these two stops, reverse that segment. A GPU can test hundreds of thousands of tweaks simultaneously, but that trick lives only in CUDA research code and NVIDIA's cuOpt product; nobody has ever made it something you can *watch*. This runs in any browser tab on your own graphics card: open the page, watch a 10,000-stop route tighten in real time, with a live counter racing the GPU against your CPU.

### The HN title test
**"Show HN: Watch your GPU test 200,000 route improvements per frame, in your browser"**

### Novelty evidence
- [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) — open-sourced 2025, claims ~100× local-search speedups, but CUDA-only; its "web demo" submits jobs to a hosted GPU backend — the browser computes nothing.
- [or-tools-wasm](https://github.com/Axelwickm/or-tools-wasm) — real routing solver in the browser, but CPU/WASM, request-response, no live search visualization.
- [tspvis](https://github.com/jhackshaw/tspvis) — the visual prior art; verifier confirmed from source it runs Web Workers on CPU, no GPU anywhere.
- Academic CUDA 2-opt repos/papers (2011→2026, e.g. cuGenOpt) — prove the parallel-move-evaluation + reduction pattern works; 100% CUDA, zero browser.
- Verifier ran "WebGPU TSP", "WebGPU 2-opt", "GPU route optimization browser", WebGL/GPU.js sweeps: **zero hits** at the GPU × browser × watchable intersection.

**Daylight in one sentence:** GPU routing local search is charted territory in CUDA and productized in cuOpt, but the vendor-neutral, zero-install, watch-it-happen version verifiably does not exist — and cuOpt's own hosted demo structurally can't make the "runs on *your* hardware" claim.

### Build plan
**Stack:** Rust + wgpu/WGSL (move-evaluation kernels for 2-opt/Or-opt, hierarchical per-workgroup reductions to pick winning moves), on-the-fly distance computation past ~3k nodes (a 10k dense matrix is 400MB — over buffer limits; computing distances in-shader is the standard fix), canvas/WebGL tour rendering, deck.gl for map instances.

- **Milestone 0 (one weekend):** 2-opt kernel + reduction on TSPLIB instances (berlin52 → pr2392), animated tour, moves-evaluated-per-second counter, CPU-vs-GPU race toggle.
- **Milestone 1 (benchmarkable):** table of gap-to-known-optimal across TSPLIB and CVRPLIB X-instances at fixed wall-clock (30s/60s) on named consumer hardware (M-series Mac, mid-range AMD laptop), vs. or-tools-wasm on the same machine — "X% gap in N seconds, client-side, integrated graphics" is the falsifiable claim.
- **Milestone 2 (shareable artifact):** polished site with the 10,000-stop live demo + CVRP support + writeup on WebGPU move-conflict resolution (the technically meaty post).

**Data:** TSPLIB — http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/ (verified live at root; instance subdirectory worth a manual click). CVRPLIB — https://galgos.inf.puc-rio.br/cvrplib/index.php/en/ (verified live at its new mirror, downloads confirmed, includes the 2026 XL set up to 10k+ customers). SINTEF Solomon/Li&Lim — https://www.sintef.no/projectweb/top/ (verified live).

### Kill risks
1. **"Nice demo" ceiling:** the honest contribution is accessibility, not algorithms; if the framing drifts toward performance claims, "cuOpt does this 100× faster on an H100" is a fair dunk waiting in the comments.
2. **WebGPU engineering rabbit holes:** lock-free parallel move commits and cross-vendor atomics quirks could eat weeks; the CUDA papers chart the pattern but WGSL portability is the unknown.
3. **Thematic rerun:** this is the builder's third routing project — motivation risk, and sophisticated readers may notice the substrate changed more than the ideas.

### Rubric scores
- **Legibility: 5** — a route on a map getting shorter while a counter spins; nothing to explain.
- **Novelty: 3** — the verified daylight is real but presentational: known technique, new venue (browser/any-GPU/watchable).
- **Demoability: 5** — the single most demoable candidate in the entire survey.
- **Benchmark truth: 5** — TSPLIB/CVRPLIB gaps-to-best-known are checkable by anyone, in their own browser, on the same hardware class.
- **Right-sized: 4** — the move-eval/reduce pattern is proven in CUDA and WebGPU handles comparable particle workloads at 60fps; conflict resolution is fiddly but bounded.
- **HN resonance: 5** — the watch-it-happen pattern is HN's most durable optimization cluster (484/253/232), and this adds GPU + zero-install.

**Total: 27/30**

---

## Proposal 4 — "Solverscope": a flame graph for how solvers think

### The pitch
When an optimization solver spends an hour finding the best schedule, it's exploring millions of what-if branches — and today that search is a black box you watch through a scrolling text log. This is a flame graph for that search: a live, zoomable picture in your browser of the solver's decision tree growing, getting pruned, and converging, with a scrubber to replay any solve. Every previous tool for this died in the 1990s.

### The HN title test
**"Show HN: A flame graph for optimization solvers — watch SCIP think, live"**

### Novelty evidence
- [TreeD](https://github.com/mattmilten/TreeD) — the one modern attempt; verifier confirmed via GitHub API: last push April 2022, 30 stars, static Plotly output. Dead.
- [GrUMPy](https://github.com/coin-or/GrUMPy) — the only *maintained* near-miss (pushed 2026-04-03), but by its own docs a classroom tool: toy Python B&B, static Graphviz, no streaming, no real solver at scale.
- The .vbc lineage (VBCTOOL/HyDraw/vbc2dot) — verifier fetched both SCIP mailing-list threads ([2020](https://listserv.zib.de/pipermail/scip/2020-May/003945.html), [2018](https://listserv.zib.de/pipermail/scip/2018-February/003288.html)) showing practitioners asking for exactly this and being pointed at 1990s tooling.
- [grblogtools](https://github.com/Gurobi/grblogtools) (Gurobi, pushed 2026-06) and [CP-SAT-Log-Analyzer](https://github.com/d-krupke/CP-SAT-Log-Analyzer) — both actively maintained, both visualize *aggregate metrics*, neither renders tree topology (their solvers can't expose it — which structurally protects this niche).
- [Ecole](https://github.com/ds4dm/ecole) — proves SCIP's live per-node hook is cheap and robust (an entire RL research field runs on it); nobody ever pointed it at a renderer.

**Daylight in one sentence:** no maintained tool anywhere renders a real solver's live search-tree topology in a browser, and the mechanism is verified better than expected — SCIP's event handlers expose node id/parent/depth/bound/branching decision directly, with official Rust bindings ([russcip](https://github.com/scipopt/russcip), maintained by the SCIP team, pushed 2026-06-30), and SCIP's 2022 relicense to Apache 2.0 makes open tooling on its internals legally clean for the first time.

### Build plan
**Stack:** Rust — russcip event handler streaming NDJSON over WebSocket; browser front-end on mature MIT rendering libs (d3-flame-graph / zoomable icicle) with level-of-detail aggregation; replay format for finished solves. SCIP-only by design (verifier ruled out HiGHS — callbacks expose only aggregate counts — and CP-SAT — portfolio solver, no single tree).

- **Milestone 0 (one weekend):** event handler → WebSocket → live icicle on a small MIPLIB instance; scrubber over a recorded log; bound-vs-time chart synced to the tree.
- **Milestone 1 (benchmarkable):** handle 100k+-node instances live (LOD/subtree coalescing) and million-node instances in replay mode; publish a "tree zoo" — recorded, annotated solves of ~20 curated MIPLIB instances showing recognizable pathologies (diving spikes, symmetry thrashing, bound stalls).
- **Milestone 2 (shareable artifact):** `cargo install solverscope` + hosted replay gallery + a writeup diagnosing two real instances from their tree shapes — proof it's a debugger, not eye candy.

**Data:** MIPLIB 2017 — https://miplib.zib.de (verified live, solution-file update logged Jan 2026; instances freely downloadable; the curated 240-instance benchmark set defines "easy" instances that solve in minutes — ideal demo material).

### Kill risks
1. **SCIP-only audience:** the two trendiest open solvers verifiably can't feed it, so the user base is SCIP's research/practitioner community — thousands, not millions (the flame-graph analogy's pre-mainstream tier).
2. **Million-node streaming is a real systems problem:** live mode may cap at medium instances; if replay mode carries the demo, "live" softens to "replayable."
3. **Eye-candy critique:** without the 2–3 concrete diagnosis stories, practitioners will ask what decision it changed; the tree zoo isn't optional.

### Rubric scores
- **Legibility: 5** — "flame graph for solvers" explains itself to every HN reader in four words.
- **Novelty: 4** — verified dead field since the 1990s with one classroom-toy exception; the daylight is precisely named.
- **Demoability: 4** — live icicle over a real solve in a weekend; needs curated instances to look great.
- **Benchmark truth: 3** — it's a tool, not a solver: MIPLIB supplies material but there's no score to beat, only performance floors (nodes/sec rendered).
- **Right-sized: 5** — official Rust bindings + mature rendering libs + a clean live/replay scope split make this the safest build in the set.
- **HN resonance: 4** — dev-tool visualization with a mesmerizing GIF; niche ceiling but strong click pattern ("watch it think" — Stockfish-internals essay: 355).

**Total: 25/30**

---

## Proposal 5 — "Whynot": compiler error messages for impossible planning problems

### The pitch
When free optimization tools can't find a valid schedule, they print the equivalent of `error` with no line number — while the $12k/year commercial solvers name the exact contradiction. This tool takes any standard model file, finds the smallest set of rules that can't all hold ("Alice needs 4 rest days" + "every shift needs 3 nurses" + "you have 9 nurses"), and explains the contradiction in plain English with a picture. Eight years of forum threads ask for exactly this.

### The HN title test
**"Show HN: A tool that tells you WHY your schedule is impossible (works with free solvers)"**

### Novelty evidence
The pain is the best-documented in the survey: [or-tools#973](https://github.com/google/or-tools/issues/973) (open since 2018), [JuMP#3034](https://github.com/jump-dev/JuMP.jl/issues/3034), 7+ OR Stack Exchange threads (2016–2024), and [or.stackexchange #5150](https://or.stackexchange.com/questions/5150) (33 votes) naming this a commercial-only advantage. But verification found the extraction layer is more built-out than the pitch assumed — which reshaped the project:

- [SCIP 10.0](https://www.zib.de/news/scip-optimization-suite-1000-released) (Nov 2025) now ships a native IIS finder, marketed as "a novel tool for explaining infeasibility" — solver-native, no narrative, no graph.
- [cpmpy.tools.explain](https://cpmpy.readthedocs.io/en/latest/api/tools/explain.html) (Apache-2.0, release June 2026) ships MUS/QuickXplain for CP-SAT and others — but you must model in its Python DSL, and output is a raw constraint list.
- [MathOptAnalyzer.jl](https://github.com/jump-dev/MathOptAnalyzer.jl) (v0.1, April 2026, experimental) — Julia-only.
- [pyomo.contrib.iis](https://pyomo.readthedocs.io/en/stable/explanation/analysis/iis.html), [MiniZinc FindMUS](https://docs.minizinc.dev/en/stable/find_mus.html) — each locked to its own modeling language.
- [OptiChat](https://github.com/li-group/OptiChat) — the one narrative attempt: Pyomo-only, requires an OpenAI key; the one graph-explanation paper (arXiv:2507.13007) is 0-star throwaway code requiring CPLEX.

**Daylight in one sentence:** conflict *extraction* now exists per-ecosystem, but nothing ingests raw MPS/LP/CP-SAT files without DSL lock-in, normalizes across backends (SCIP IIS / HiGHS getIis / cpmpy MUS), and produces the plain-English narrative + dependency graph — the layer every one of those tools stops short of, and the part users actually asked for.

### Build plan
**Stack:** Rust CLI + WASM web UI; per-backend adapters (HiGHS `getIis` — shipped, actively hardened as of v1.15.1 four days before this survey; SCIP 10 IIS finder; cpmpy/CP-SAT with `num_workers=1` pinned and verify-twice, since CP-SAT's multithreaded nondeterminism is documented); templated narrative generation first, optional local-LLM polish second; d3 conflict-graph rendering.

- **Milestone 0 (one weekend):** feed a deliberately over-constrained nurse roster to HiGHS, get the minimal conflict set, render it highlighted in the original file + as a small graph + a templated English paragraph.
- **Milestone 1 (benchmarkable):** MIPLIB's infeasible-tagged set (~44 instances, verified) for coverage/runtime; injected-conflict recovery protocol (tighten one known constraint in a feasible instance → check the tool names exactly it) for precision; publish the table.
- **Milestone 2 (shareable artifact):** drop-a-file web version + CP-SAT path + "the missing error message" writeup; propose upstreaming the narrative layer to HiGHS/OR-Tools docs.

**Data:** MIPLIB 2017 — https://miplib.zib.de (verified live; infeasible tag confirmed, ~44–45 instances). Nurse rostering — https://www.schedulingbenchmarks.org (verified live; note: XML format has parsing friction — treat as secondary).

### Kill risks
1. **Narrative quality is the product:** a wrong or vacuous English explanation is worse than a raw constraint list; mapping constraint IDs to human meaning is genuinely hard on real-world files with names like `c4821`.
2. **Plumbing eats the timebox:** three backends × immature APIs (HiGHS IIS was bug-fixed the week of this survey; CP-SAT needs determinism workarounds) is real integration risk.
3. **Modest ceiling:** dev-tool utility without visual wow — likely respected (and used) more than upvoted.

### Rubric scores
- **Legibility: 5** — "compiler error messages for planning problems" lands instantly on a developer audience.
- **Novelty: 3** — extraction exists in five ecosystems; the verified gap is the raw-file + normalization + narrative layer, which is real but integrative.
- **Demoability: 4** — the broken-roster → plain-English-contradiction demo is a weekend build and deeply satisfying, if not flashy.
- **Benchmark truth: 4** — MIPLIB's infeasible set plus the injected-conflict protocol make claims checkable, though part of the methodology is self-constructed.
- **Right-sized: 5** — a thin, well-scoped layer over native extractors; the safest scope in the set alongside Solverscope.
- **HN resonance: 4** — "better error messages" is durable dev catnip; lacks the visual hook of the others.

**Total: 25/30**

---

## Killed in verification (so you don't re-derive them)

- **Permissively-licensed LKH-class TSP solver** — premise falsified: [GA-EAX](https://github.com/nagata-yuichi/GA-EAX) is MIT-licensed and *beats* LKH on 50/57 large instances, and has sat unnoticed since 2013 (68 combined stars across forks) — the demand is revealed to be thin. Salvageable only as "finish the abandoned Rust chained-LK" packaging play.
- **Taillard job-shop record attack** — the open-instance table (19 of 80 remain open, verified at [optimizizer.com/TA.php](https://www.optimizizer.com/TA.php)) is being raced *monthly* by institutional CP-SAT and RL teams (2026-tagged submissions in the table itself), and "LLM-evolved LNS for scheduling" is already published (IJCAI 2025 NS4S; LLM-LNS, Mar 2026). Verifier's odds for a solo bound improvement: ~10%. The quieter alternative (Vallada-Ruiz-Framinan flow-shop table) is noted if you ever want a records side-quest.
- **Heuristic-overfitting stress-tester** — the "nobody tests generalization" premise is stale: EoH/ReEvo now report out-of-distribution results as standard practice, [CO-Bench](https://github.com/sunnweiwei/CO-Bench) and [HeuriGym](https://cornell-zhang.github.io/heurigym) exist as maintained multi-problem harnesses, and 2025–26 work (Robusta, DASH) already auto-*fixes* overfitting. The one-shot article version ("I re-ran DeepMind's bin-packing heuristic on 12 unseen datasets") could still be a fine blog post — it's just not a project.

Also killed at checkpoint (one line each): learning-to-branch plugin (jargon- and infra-heavy), neural-solver ROI calculator + solver-picker site (content, not builds), browser rostering (Timefold owns it), minimal-perturbation replanning (known trick), gerrymandering optimizer (GerryChain/DRA exist), nightly solver benchmark dashboard (untrusted shared hardware), "SQLite-of-MIP" Rust solver (grad-project wheel, weak daylight per its own scout), CVRPLIB XL attack (0.1%-shaving = your depth-without-legibility trap), MIPLIB open-set attack and QAPLIB Tai256c (illegible / 30-year long-shot), ROADEF 2026 entry (competition ≠ shareable artifact).

---

## Ranked comparison

| # | Proposal | Leg | Nov | Demo | Bench | Size | HN | Total | Verdict |
|---|----------|-----|-----|------|-------|------|----|-------|---------|
| 1 | **Sundial** — WebGPU LP solver | 4 | 5 | 4 | 5 | 4 | 5 | **27** | ALIVE |
| 2 | **Packbench** — verified packing registry + attack | 5 | 4 | 5 | 5 | 4 | 5 | **28** | WOUNDED→reshaped |
| 3 | **Swarmroute** — GPU route search, watchable | 5 | 3 | 5 | 5 | 4 | 5 | **27** | ALIVE |
| 4 | **Solverscope** — flame graph for solvers | 5 | 4 | 4 | 3 | 5 | 4 | **25** | ALIVE |
| 5 | **Whynot** — infeasibility error messages | 5 | 3 | 4 | 4 | 5 | 4 | **25** | WOUNDED→reshaped |

**On the ranking:** Packbench has the highest raw score but sits at #2 — the totals treat all dimensions equally, and the tiebreak that matters is *whose hands the headline outcome is in*. Sundial's risks are entirely execution risks (numerics, buffer limits) under your control, and its novelty claim — first LP solver on WebGPU, ever — cannot be scooped quietly in 8 weeks and survives even a mediocre benchmark table. Packbench's best story (a new record) depends partly on racing active defenders; its floor (registry + auditor) is genuinely shippable but softer. Swarmroute matches Sundial's total with a thinner novelty claim and a thematic-rerun risk — and notably, it shares ~60% of its infrastructure (WGSL kernels, reductions, Rust/wgpu scaffolding) with Sundial, making it a natural *sequel* rather than a competitor.

## Top recommendation: **Sundial (Proposal 1)**

It is the only candidate where two independent search sweeps found *zero* prior art at the core claim; the "why now" sharpened three months ago when HiGHS committed its GPU future to CUDA; the benchmark (updated five days ago) already contains the CUDA baselines to measure against; the f32 objection that could have killed it is pre-answered by the reference implementation's own single-precision mode; and its failure mode still ships a first-of-kind, genuinely useful artifact with an honest number attached. It hits your taste stack almost perfectly — Rust, browser, GPU, benchmark-checkable — and its HN pitch ("any GPU, no CUDA, no install") rides the strongest anti-lock-in current in 2026 computing. Build Sundial; keep Swarmroute as the sequel on the same engine, and Packbench as the discovery-itch project if a record hunt ever calls.
