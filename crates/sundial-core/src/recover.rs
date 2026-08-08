//! Integral matching recovery for the taxi demo.
//!
//! The H3-scaled matching LP certifies `Optimal` on the GPU (~6 s full scale),
//! but on the dense real fixture the optimal *face* is high-dimensional
//! (real-data ties + 4-decimal rounding ⇒ hundreds of near-equidistant riders),
//! so the objective-converged fractional plan does not read off a crisp pairing
//! (only ~88/1024 riders are single-cab, measured on the real fixture). This
//! module rounds that fractional plan to a genuine
//! injective rider→cab assignment.
//!
//! **Honesty contract.** `recover_matching` makes NO optimality claim. The
//! claim is made by *comparison*: callers report the returned `total_cost`
//! against `certified_floor` (a rigorous CPU-f64 lower bound on the matching
//! optimum via a repaired + coordinate-ascent-tightened feasible dual, in the
//! same coordinate units) and phrase display copy from the measured, always
//! non-negative slack. `recover_matching` itself is not certificate-grade; the
//! floor is (weak duality, feasible by construction).
//!
//! Algorithm:
//!   1. Sparse candidate graph = LP-support edges (entries above a small
//!      threshold relative to the rider's row mass) ∪ the k = 8 nearest cabs
//!      per rider. The kNN union guarantees the graph always admits a
//!      rider-perfect matching.
//!   2. Exact min-cost rider-perfect matching on that graph via successive
//!      shortest augmenting paths with Johnson potentials (Dijkstra on
//!      reduced costs). All costs ≥ 0, so potentials start at 0. Deterministic:
//!      adjacency built in fixed order, Dijkstra ties broken by node index.
//!   3. Geometric uncrossing sweep: while any two rider→cab segments *properly*
//!      cross (open-segment intersection, exact-enough f64 orientation), swap
//!      their cab endpoints. A proper crossing forces both triangle
//!      inequalities strict (the crossing point is interior and non-collinear),
//!      so each swap strictly decreases total metric cost ⇒ the sweep
//!      terminates. Zero crossings are asserted at exit.
//!
//! Determinism: identical inputs ⇒ identical output (no RNG, fixed iteration
//! orders, index tie-breaks).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Nearest cabs unioned into every rider's candidate set; the union guarantees
/// a rider-perfect matching exists on the graph even if the LP support is thin.
/// k = 8 is a fixed contract, not a tuning knob: the tests that prove exact
/// recovery on small instances depend on this value.
const KNN: usize = 8;
/// LP-support inclusion threshold, relative to the rider's row mass: entries
/// this small are numerical dust from the f32 GPU iterate, not real support.
const SUPPORT_REL: f64 = 1e-3;

#[derive(Debug, Clone)]
pub struct RecoveredMatching {
    /// `assignment[i]` = cab index served to rider `i`; injective (each rider
    /// gets a distinct cab).
    pub assignment: Vec<u32>,
    /// Σ Euclidean rider→cab distance over the recovered assignment, in the
    /// callers' input coordinate units (mass-1, i.e. NOT the `1/nt`-scaled LP
    /// objective).
    pub total_cost: f64,
    /// Number of LP-support edges (plan entries above the relative threshold) —
    /// a diagnostic of how fractional the solved plan was, not the graph size.
    pub support_edges: usize,
}

#[inline]
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// f64 orientation of `c` relative to the directed line `a→b`: >0 left, <0
/// right, 0 collinear. Plain cross product — no error-free transform needed,
/// this runs on the CPU (never on Metal fast-math).
#[inline]
fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Do segments `a→b` and `c→d` *properly* cross (open-segment intersection)?
/// Requires strict sign changes on both orientation triples; any collinear or
/// endpoint-touching configuration (some orientation exactly 0) is treated as
/// NON-crossing. Proper crossing ⇒ the intersection point is strictly interior
/// and no tested triple is collinear ⇒ swapping cab endpoints strictly
/// decreases total length (the termination guarantee for the sweep).
pub fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let d1 = orient(a, b, c);
    let d2 = orient(a, b, d);
    let d3 = orient(c, d, a);
    let d4 = orient(c, d, b);
    (d1 * d2 < 0.0) && (d3 * d4 < 0.0)
}

/// Default dual-ascent sweep cap for `certified_floor`.
pub const FLOOR_MAX_SWEEPS: usize = 100;

/// Result of the certified-floor computation with ascent diagnostics.
#[derive(Debug, Clone, Copy)]
pub struct CertifiedFloor {
    /// Rigorous lower bound on the (unscaled, mass-1) matching optimum.
    pub value: f64,
    /// Dual-ascent sweeps actually performed (≤ the requested cap; fewer if the
    /// floor converged early).
    pub sweeps: usize,
}

/// Rigorous CPU-f64 certified lower bound on the (unscaled, mass-1) matching
/// optimum, via a REPAIRED + coordinate-ascent-tightened feasible dual (weak
/// duality). The value is ≤ the true optimum BY CONSTRUCTION at every iterate —
/// independent of the solver's tolerance — so it is a genuine floor, not the
/// tolerance-dependent readout `dual_obj × nt` (which at tol 1e-4 overshoots the
/// optimum; see task-4a-report.md). This is the diagnostic form returning the
/// sweep count; `certified_floor` wraps it for the plain `f64`.
///
/// `y` is the solver-returned dual in the `OpProblem`'s own row space:
/// `y[0..ns]` = rider equality rows, `y[ns..ns+nt]` = cab capacity rows.
/// `rider_mass` / `cab_cap` are the H3-scaled masses (both `1/nt`). The bound is
/// formed in scaled space and rescaled to unscaled distance units (× nt,
/// consistent with `total_cost = primal_obj × nt`), so it is directly comparable
/// to `RecoveredMatching::total_cost`.
///
/// LP dual of `min Σ c_ij x_ij` s.t. rider rows `Σ_j x_ij = m_i` (dual `u_i`
/// free), cab rows `Σ_i x_ij ≤ cap_j` (dual `v_j ≤ 0`), `x ≥ 0`: dual feasibility
/// is `u_i + v_j ≤ c_ij` ∀i,j; the objective is `max Σ m_i u_i + Σ cap_j v_j`.
/// The solver's reduced-cost convention is `c_ij + y_i + y_{ns+j} ≥ 0` (see
/// `kkt::residuals_view`), i.e. `v_j = −y_{ns+j}`.
///
/// Steps: (init) `v_j ← min(−y_{ns+j}, 0)`, then `u_i ← min_j(c_ij − v_j)`;
/// (ascent) alternate the two exact coordinate-wise maximizers —
/// `v_j ← min(0, min_i(c_ij − u_i))` then `u_i ← min_j(c_ij − v_j)`. Each
/// half-step keeps `(u, v)` feasible and never decreases the objective, so the
/// bound stays rigorous every sweep and only tightens. Iterate up to
/// `max_sweeps`, stopping early once a sweep gains `< 1e-9·(1+|floor|)`.
///
/// The rider duals `y[0..ns]` are NOT trusted — `u` is reconstructed from `v`.
/// The distance matrix is recomputed fresh in f64 from the input coordinates at
/// entry (no external/cached cost trusted) and reused across sweeps.
pub fn certified_floor_ascent(
    y: &[f64],
    riders: &[[f64; 2]],
    cabs: &[[f64; 2]],
    rider_mass: f64,
    cab_cap: f64,
    max_sweeps: usize,
) -> CertifiedFloor {
    let ns = riders.len();
    let nt = cabs.len();
    assert_eq!(y.len(), ns + nt, "dual length must be ns + nt");

    // Fresh f64 distance matrix (row-major), recomputed from coordinates — no
    // external/cached cost trusted. Reused across sweeps so each is O(ns·nt)
    // min-reductions with no repeated sqrt. Transient footprint: ns·nt·8 bytes.
    let mut dmat = vec![0.0f64; ns * nt];
    for (i, r) in riders.iter().enumerate() {
        let row = &mut dmat[i * nt..(i + 1) * nt];
        for (j, k) in cabs.iter().enumerate() {
            row[j] = dist(*r, *k);
        }
    }

    // u ← min_j (dmat_ij − v_j), row-sequential over dmat.
    let update_u = |dmat: &[f64], v: &[f64], u: &mut [f64]| {
        for (i, u_i) in u.iter_mut().enumerate() {
            let row = &dmat[i * nt..(i + 1) * nt];
            let mut m = f64::INFINITY;
            for j in 0..nt {
                let cand = row[j] - v[j];
                if cand < m {
                    m = cand;
                }
            }
            *u_i = m;
        }
    };
    // v_j ← min(0, min_i (dmat_ij − u_i)), accumulated row-sequential over dmat.
    let update_v = |dmat: &[f64], u: &[f64], v: &mut [f64]| {
        for vj in v.iter_mut() {
            *vj = f64::INFINITY;
        }
        for (i, &u_i) in u.iter().enumerate() {
            let row = &dmat[i * nt..(i + 1) * nt];
            for (j, vj) in v.iter_mut().enumerate() {
                let cand = row[j] - u_i;
                if cand < *vj {
                    *vj = cand;
                }
            }
        }
        for vj in v.iter_mut() {
            if *vj > 0.0 {
                *vj = 0.0;
            }
        }
    };
    // Coordinate-unit floor Σ m_i u_i + Σ cap_j v_j, × nt (see doc: under H3
    // rider_mass = cab_cap = 1/nt ⇒ × nt gives the mass-1 bound Σ u + Σ v).
    let nt_f = nt as f64;
    let coord_floor = |u: &[f64], v: &[f64]| -> f64 {
        (rider_mass * u.iter().sum::<f64>() + cab_cap * v.iter().sum::<f64>()) * nt_f
    };

    // init: repair v from y, then the largest feasible u.
    let mut v: Vec<f64> = (0..nt).map(|j| (-y[ns + j]).min(0.0)).collect();
    let mut u = vec![0.0f64; ns];
    update_u(&dmat, &v, &mut u);
    let mut floor = coord_floor(&u, &v);

    // ascent: alternate exact block maximizers, monotone + feasible every step.
    let mut sweeps = 0;
    for _ in 0..max_sweeps {
        update_v(&dmat, &u, &mut v);
        update_u(&dmat, &v, &mut u);
        sweeps += 1;
        let new_floor = coord_floor(&u, &v);
        let gained = new_floor - floor;
        floor = new_floor;
        if gained < 1e-9 * (1.0 + new_floor.abs()) {
            break;
        }
    }

    CertifiedFloor {
        value: floor,
        sweeps,
    }
}

/// Convenience wrapper: `certified_floor_ascent` with the default sweep cap,
/// returning just the rigorous bound. This is the stable call-site API.
pub fn certified_floor(
    y: &[f64],
    riders: &[[f64; 2]],
    cabs: &[[f64; 2]],
    rider_mass: f64,
    cab_cap: f64,
) -> f64 {
    certified_floor_ascent(y, riders, cabs, rider_mass, cab_cap, FLOOR_MAX_SWEEPS).value
}

/// Total-order key over f64 distances for the Dijkstra heap (f64 is not `Ord`).
#[derive(PartialEq)]
struct Key(f64);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Key {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Minimal residual-graph min-cost flow (unit capacities, non-negative arc
/// costs). Edges are stored in adjacency lists of edge ids; each `add` pushes a
/// forward edge at an even index and its reverse at the odd successor, so
/// `e ^ 1` toggles between them.
struct Mcmf {
    to: Vec<usize>,
    cap: Vec<i64>,
    cost: Vec<f64>,
    adj: Vec<Vec<usize>>,
}

impl Mcmf {
    fn new(nodes: usize) -> Self {
        Mcmf {
            to: Vec::new(),
            cap: Vec::new(),
            cost: Vec::new(),
            adj: vec![Vec::new(); nodes],
        }
    }

    /// Add a directed edge `u→v` (cap, cost) plus its zero-cap reverse.
    /// Returns the forward edge id.
    fn add(&mut self, u: usize, v: usize, cap: i64, cost: f64) -> usize {
        let e = self.to.len();
        self.to.push(v);
        self.cap.push(cap);
        self.cost.push(cost);
        self.adj[u].push(e);
        self.to.push(u);
        self.cap.push(0);
        self.cost.push(-cost);
        self.adj[v].push(e + 1);
        e
    }
}

/// Exact min-cost rider-perfect matching on the candidate graph via successive
/// shortest augmenting paths with Johnson potentials. `cand[i]` lists candidate
/// cab indices for rider `i` (in fixed order). Returns `assignment[i] = cab`.
/// Panics if some rider cannot be matched (the graph must contain a perfect
/// matching by construction — kNN ∪ support).
fn min_cost_matching(riders: &[[f64; 2]], cabs: &[[f64; 2]], cand: &[Vec<usize>]) -> Vec<u32> {
    let ns = riders.len();
    let nt = cabs.len();
    let s = 0;
    let t = ns + nt + 1;
    let nodes = ns + nt + 2;
    let mut g = Mcmf::new(nodes);

    // S → rider  (cap 1, cost 0)
    for i in 0..ns {
        g.add(s, 1 + i, 1, 0.0);
    }
    // cab → T    (cap 1, cost 0)
    for j in 0..nt {
        g.add(1 + ns + j, t, 1, 0.0);
    }
    // rider → cab (cap 1, cost = distance). Remember the forward edge id per
    // rider so we can read the chosen cab after the flow settles.
    let mut rider_edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ns]; // (cab j, edge id)
    for i in 0..ns {
        for &j in &cand[i] {
            let e = g.add(1 + i, 1 + ns + j, 1, dist(riders[i], cabs[j]));
            rider_edges[i].push((j, e));
        }
    }

    let mut h = vec![0.0f64; nodes]; // potentials; all arc costs ≥ 0 ⇒ start 0
    let mut d = vec![f64::INFINITY; nodes];
    let mut prev_edge = vec![usize::MAX; nodes];

    for _ in 0..ns {
        for x in d.iter_mut() {
            *x = f64::INFINITY;
        }
        d[s] = 0.0;
        let mut heap: BinaryHeap<std::cmp::Reverse<(Key, usize)>> = BinaryHeap::new();
        heap.push(std::cmp::Reverse((Key(0.0), s)));
        while let Some(std::cmp::Reverse((Key(du), u))) = heap.pop() {
            if du > d[u] {
                continue;
            }
            for &e in &g.adj[u] {
                if g.cap[e] <= 0 {
                    continue;
                }
                let v = g.to[e];
                // reduced cost is ≥ 0 by the potential invariant
                let nd = du + g.cost[e] + h[u] - h[v];
                if nd + 1e-15 < d[v] {
                    d[v] = nd;
                    prev_edge[v] = e;
                    heap.push(std::cmp::Reverse((Key(nd), v)));
                }
            }
        }
        assert!(
            d[t].is_finite(),
            "candidate graph has no rider-perfect matching (should be impossible: kNN ∪ support)"
        );
        // lift potentials by the settled distances (reachable nodes only)
        for node in 0..nodes {
            if d[node].is_finite() {
                h[node] += d[node];
            }
        }
        // augment one unit of flow along the shortest S→T path
        let mut node = t;
        while node != s {
            let e = prev_edge[node];
            g.cap[e] -= 1;
            g.cap[e ^ 1] += 1;
            node = g.to[e ^ 1]; // the tail of forward edge e
        }
    }

    // read the matching: the rider→cab forward edge that carries flow has cap 0
    let mut assignment = vec![u32::MAX; ns];
    for i in 0..ns {
        for &(j, e) in &rider_edges[i] {
            if g.cap[e] == 0 {
                assignment[i] = j as u32;
                break;
            }
        }
        assert_ne!(assignment[i], u32::MAX, "rider {i} left unmatched");
    }
    assignment
}

/// Geometric uncrossing sweep. Repeatedly scans all rider pairs in fixed order;
/// when two segments properly cross, swaps their cab endpoints (a strictly
/// cost-decreasing move). Passes are capped defensively; zero crossings are
/// asserted at exit.
fn uncross(assignment: &mut [u32], riders: &[[f64; 2]], cabs: &[[f64; 2]]) {
    let n = assignment.len();
    // Strict cost decrease per swap bounds the total swap count; in practice a
    // handful of passes suffice. Cap generously and assert convergence.
    let max_passes = 8 * n + 64;
    let mut converged = false;
    for _ in 0..max_passes {
        let mut swapped = false;
        for a in 0..n {
            for b in (a + 1)..n {
                let ra = riders[a];
                let rb = riders[b];
                let pa = cabs[assignment[a] as usize];
                let pb = cabs[assignment[b] as usize];
                if segments_cross(ra, pa, rb, pb) {
                    assignment.swap(a, b);
                    swapped = true;
                }
            }
        }
        if !swapped {
            converged = true;
            break;
        }
    }
    assert!(
        converged,
        "uncrossing sweep did not converge within {max_passes} passes"
    );
    // defensive exhaustive re-check: no proper crossings remain
    for a in 0..n {
        for b in (a + 1)..n {
            let pa = cabs[assignment[a] as usize];
            let pb = cabs[assignment[b] as usize];
            debug_assert!(
                !segments_cross(riders[a], pa, riders[b], pb),
                "crossing survived the sweep"
            );
        }
    }
}

/// Recover an injective rider→cab assignment from a solved (fractional) plan.
/// `x` is the `ns·nt` transport plan (`x[i·nt + j]`), `riders`/`cabs` the input
/// point clouds. Makes NO optimality claim (see module docs / honesty contract).
pub fn recover_matching(x: &[f64], riders: &[[f64; 2]], cabs: &[[f64; 2]]) -> RecoveredMatching {
    let ns = riders.len();
    let nt = cabs.len();
    assert_eq!(x.len(), ns * nt, "plan length must be ns·nt");
    assert!(nt >= ns, "need cabs ≥ riders");

    // 1. Candidate graph: LP-support edges ∪ kNN cabs per rider.
    let mut support_edges = 0usize;
    let mut cand: Vec<Vec<usize>> = Vec::with_capacity(ns);
    for i in 0..ns {
        let row = &x[i * nt..(i + 1) * nt];
        let row_mass: f64 = row.iter().copied().map(|v| v.max(0.0)).sum();
        let thresh = SUPPORT_REL * row_mass;

        let mut in_set = vec![false; nt];
        let mut edges: Vec<usize> = Vec::new();
        // LP-support edges (above the relative threshold)
        for (j, &v) in row.iter().enumerate() {
            if v > thresh && v > 0.0 {
                support_edges += 1;
                if !in_set[j] {
                    in_set[j] = true;
                    edges.push(j);
                }
            }
        }
        // union the k nearest cabs (by distance, index tie-break)
        let mut order: Vec<usize> = (0..nt).collect();
        order.sort_by(|&p, &q| {
            let dp = dist(riders[i], cabs[p]);
            let dq = dist(riders[i], cabs[q]);
            dp.total_cmp(&dq).then(p.cmp(&q))
        });
        for &j in order.iter().take(KNN.min(nt)) {
            if !in_set[j] {
                in_set[j] = true;
                edges.push(j);
            }
        }
        edges.sort_unstable(); // fixed order ⇒ deterministic matching
        cand.push(edges);
    }

    // 2. Exact min-cost rider-perfect matching on the candidate graph.
    let mut assignment = min_cost_matching(riders, cabs, &cand);

    // 3. Geometric uncrossing polish.
    uncross(&mut assignment, riders, cabs);

    let total_cost: f64 = assignment
        .iter()
        .enumerate()
        .map(|(i, &j)| dist(riders[i], cabs[j as usize]))
        .sum();

    RecoveredMatching {
        assignment,
        total_cost,
        support_edges,
    }
}
