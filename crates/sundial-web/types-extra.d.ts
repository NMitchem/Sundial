// Result-shape declarations for sundial-lp (hand-maintained; the generated
// bindings type results as `any` because they cross serde_wasm_bindgen).
export interface SundialResult {
  status: string; // "Optimal (CPU f64 verified)" | "IterationLimit" | "Infeasible" | …
  objective: number;
  iterations: number;
  restarts: number;
  solve_ms: number;
  rel_primal: number;
  rel_dual: number;
  rel_gap: number;
  adapter: string;
  n_vars: number;
}
export interface TransportPreview {
  grid: number;
  src: number[];
  tgt: number[];
}
export interface MatchResult extends SundialResult {
  assignment: number[]; // per-rider assigned cab (integral, injective, crossing-free)
  total_cost: number; // recovered matching total distance, coordinate units
  certified_floor: number; // rigorous lower bound on any dispatch, coordinate units
  support_edges: number;
}
