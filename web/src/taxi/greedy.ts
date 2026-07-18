import { Pt } from "./data";

export function dist(a: Pt, b: Pt): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** The baseline every engineer would write: riders in order, each takes the
 *  nearest still-free cab. O(nr·nc) — instant at demo scale. */
export function greedyAssign(riders: Pt[], cabs: Pt[]): Uint32Array {
  const taken = new Uint8Array(cabs.length);
  const out = new Uint32Array(riders.length);
  riders.forEach((r, i) => {
    let best = -1;
    let bestD = Infinity;
    for (let j = 0; j < cabs.length; j++) {
      if (taken[j]) continue;
      const d = dist(r, cabs[j]);
      if (d < bestD) {
        bestD = d;
        best = j;
      }
    }
    taken[best] = 1;
    out[i] = best;
  });
  return out;
}

export function totalMiles(
  riders: Pt[],
  cabs: Pt[],
  assign: ArrayLike<number>,
  milesPerUnit: number,
): number {
  let t = 0;
  for (let i = 0; i < riders.length; i++) t += dist(riders[i], cabs[assign[i]]);
  return t * milesPerUnit;
}
