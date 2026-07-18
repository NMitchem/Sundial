export type Pt = [number, number];

export interface TaxiPoints {
  source: string;
  window: string;
  aspect: number; // width / height of the map extent
  miles_per_unit: number; // miles per normalized coordinate unit
  riders: Pt[];
  cabs: Pt[];
  basemap: Pt[];
}

export async function loadPoints(): Promise<TaxiPoints> {
  const res = await fetch("taxi/points.json");
  if (!res.ok) throw new Error(`points.json: HTTP ${res.status}`);
  return res.json();
}
