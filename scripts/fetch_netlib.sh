#!/usr/bin/env bash
# Fetch netlib LP instances (emps-compressed). Usage:
#   scripts/fetch_netlib.sh [list_file] [subdir]
# Defaults: scripts/netlib_m1.txt data  →  bench/netlib/
# Infeasible set: scripts/fetch_netlib.sh scripts/netlib_infeas.txt infeas  →  bench/infeas/
set -euo pipefail
cd "$(dirname "$0")/.."
LIST="${1:-scripts/netlib_m1.txt}"
SUBDIR="${2:-data}"
DEST="bench/netlib"
if [ "$SUBDIR" != "data" ]; then DEST="bench/$SUBDIR"; fi
mkdir -p "$DEST" target/netlib-raw
if [ ! -x target/emps ]; then
  curl -fsSL http://www.netlib.org/lp/data/emps.c -o target/emps.c
  cc -O2 -o target/emps target/emps.c
fi
while read -r name; do
  [ -z "$name" ] && continue
  out="$DEST/$name.mps"
  if [ -f "$out" ]; then echo "have    $name"; continue; fi
  curl -fsSL "http://www.netlib.org/lp/$SUBDIR/$name" -o "target/netlib-raw/$name"
  target/emps < "target/netlib-raw/$name" > "$out"
  echo "fetched $name"
done < "$LIST"
echo "done: $(ls "$DEST" | wc -l) instances in $DEST/"
