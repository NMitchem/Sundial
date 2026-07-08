#!/usr/bin/env bash
# Fetch the M1 Netlib sweep set. netlib.org serves instances in its "emps"
# self-expanding format; netlib also serves emps.c, which we compile once.
# Instances land in bench/netlib/ (gitignored — reproducible, not vendored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bench/netlib target/netlib-raw
if [ ! -x target/emps ]; then
  curl -fsSL http://www.netlib.org/lp/data/emps.c -o target/emps.c
  cc -O2 -o target/emps target/emps.c
fi
while read -r name; do
  [ -z "$name" ] && continue
  out="bench/netlib/$name.mps"
  if [ -f "$out" ]; then echo "have    $name"; continue; fi
  curl -fsSL "http://www.netlib.org/lp/data/$name" -o "target/netlib-raw/$name"
  target/emps < "target/netlib-raw/$name" > "$out"
  echo "fetched $name"
done < scripts/netlib_m1.txt
echo "done: $(ls bench/netlib | wc -l) instances in bench/netlib/"
