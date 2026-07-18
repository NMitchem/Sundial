#!/usr/bin/env python3
"""Build web/public/taxi/points.json from the NYC TLC yellow-cab record.

June 2015 is deliberate: its files carry exact pickup/dropoff GPS
coordinates. NYC TLC's 2022 Parquet migration re-encoded the ENTIRE
historical archive (including 2015) to zone IDs only, so the official
CloudFront host no longer serves GPS for any month. This script fetches a
community-preserved mirror of the pre-migration official file (bz2 CSV, GPS
columns intact). The committed points.json is the canonical artifact; the
mirror URL is best-effort reproducibility.

Riders = pickups in a 6-minute Friday-rush window; free cabs = drop-offs in
the preceding 12 minutes (a stated proxy, disclosed on the page).

Stdlib only (urllib, bz2, csv). Run from the repo root:
    python3 scripts/fetch_taxi.py
Downloads ~284 MiB once into target/taxi-raw/ (gitignored), then streams
the ~12M-row CSV — expect a few minutes total.
"""
import bz2
import csv
import hashlib
import json
import math
import os
import urllib.request

URL = (
    "https://nyc-tlc-trip-records-pds.s3.amazonaws.com/"
    "csv/year=2015/month=06/color=yellow/yellow_tripdata_2015-06.csv.bz2"
)
CACHE = "target/taxi-raw/yellow_tripdata_2015-06.csv.bz2"
LON = (-74.03, -73.92)  # Manhattan-ish bbox
LAT = (40.70, 40.88)
RIDERS, CABS, BASEMAP = 1024, 1152, 20000
DAY = "2015-06-05"  # a Friday
RIDER_T = (DAY + " 18:00:00", DAY + " 18:05:59")
CAB_T = (DAY + " 17:48:00", DAY + " 17:59:59")

if not os.path.exists(CACHE):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    print(f"downloading {URL} -> {CACHE} (~284 MiB, one-time)...")
    urllib.request.urlretrieve(URL, CACHE + ".part")
    os.replace(CACHE + ".part", CACHE)


def in_box(lon, lat):
    return LON[0] <= lon <= LON[1] and LAT[0] <= lat <= LAT[1]


riders_rows, cabs_rows, base_rows = [], [], []
with bz2.open(CACHE, "rt", newline="") as f:
    reader = csv.DictReader(f)
    cols = {c.strip(): c for c in reader.fieldnames}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise SystemExit(f"none of {names} in header: {reader.fieldnames}")

    pu_t = col("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime")
    do_t = col("tpep_dropoff_datetime", "lpep_dropoff_datetime", "dropoff_datetime")
    pu_lon, pu_lat = col("pickup_longitude"), col("pickup_latitude")
    do_lon, do_lat = col("dropoff_longitude"), col("dropoff_latitude")
    for row in reader:
        # timestamps are fixed "YYYY-MM-DD HH:MM:SS": string compare works
        pt, dt = row[pu_t], row[do_t]
        if not (pt.startswith(DAY) or dt.startswith(DAY)):
            continue
        try:
            plon, plat = float(row[pu_lon]), float(row[pu_lat])
            dlon, dlat = float(row[do_lon]), float(row[do_lat])
        except ValueError:
            continue
        if RIDER_T[0] <= pt <= RIDER_T[1] and in_box(plon, plat):
            riders_rows.append((pt, plon, plat))
        if CAB_T[0] <= dt <= CAB_T[1] and in_box(dlon, dlat):
            cabs_rows.append((dt, dlon, dlat))
        if dt.startswith(DAY) and in_box(dlon, dlat):
            base_rows.append((dlon, dlat))

riders_rows.sort(key=lambda r: (r[0], r[1]))
cabs_rows.sort(key=lambda r: (r[0], r[1]))
assert len(riders_rows) >= RIDERS, f"only {len(riders_rows)} riders in window"
assert len(cabs_rows) >= CABS, f"only {len(cabs_rows)} cabs in window"
riders = [(r[1], r[2]) for r in riders_rows[:RIDERS]]
cabs = [(r[1], r[2]) for r in cabs_rows[:CABS]]


def stable_hash(lon, lat):
    key = f"{round(lon * 1e6)}:{round(lat * 1e6)}".encode()
    return hashlib.md5(key).hexdigest()


base_rows.sort(key=lambda r: stable_hash(*r))
basemap = base_rows[:BASEMAP]

# Equirectangular projection to miles; screen y grows DOWN. Normalize by the
# larger span so y in [0,1], x in [0, aspect]; miles_per_unit undoes it.
K = math.cos(math.radians(40.79))
MI_PER_DEG = 69.17
span_x = (LON[1] - LON[0]) * K * MI_PER_DEG
span_y = (LAT[1] - LAT[0]) * MI_PER_DEG
norm = max(span_x, span_y)


def pts(rows):
    return [
        [
            round((lon - LON[0]) * K * MI_PER_DEG / norm, 4),
            round((LAT[1] - lat) * MI_PER_DEG / norm, 4),
        ]
        for lon, lat in rows
    ]


out = {
    "source": (
        "NYC TLC yellow-cab trip records, 2015-06 "
        "(public; via community mirror of the pre-2022 GPS-era files)"
    ),
    "window": "Friday 2015-06-05 18:00-18:06; free cabs = drop-offs 17:48-18:00",
    "aspect": round(span_x / span_y, 4),
    "miles_per_unit": round(norm, 3),
    "riders": pts(riders),
    "cabs": pts(cabs),
    "basemap": pts(basemap),
}
path = "web/public/taxi/points.json"
with open(path, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(
    f"wrote {path}: {len(out['riders'])} riders, {len(out['cabs'])} cabs, "
    f"{len(out['basemap'])} basemap pts, {out['miles_per_unit']} mi/unit, "
    f"aspect {out['aspect']}"
)
