import { loadPoints, TaxiPoints } from "./data";
import { MapView } from "./map";

const $ = (id: string) => document.getElementById(id)!;

async function boot() {
  if (!("gpu" in navigator)) {
    ($("nogpu") as HTMLElement).style.display = "block";
    ($("dispatch") as HTMLButtonElement).disabled = true;
    return;
  }
  const pts: TaxiPoints = await loadPoints();
  const view = new MapView(
    $("base") as HTMLCanvasElement,
    $("dots") as HTMLCanvasElement,
    $("lines") as HTMLCanvasElement,
    pts.aspect,
    Math.min(720, Math.round(window.innerHeight * 0.78)),
  );
  view.renderBasemap(pts.basemap);
  view.renderDots(pts.riders, pts.cabs);
  $("phase").textContent = "Press DISPATCH.";
}

boot().catch((err) => {
  $("phase").textContent = `Failed to load: ${err}`;
  console.error(err);
});
