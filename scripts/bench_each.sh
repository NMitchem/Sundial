#!/usr/bin/env bash
# Resumable per-instance bench: one `sundial bench` per single-file dir so
# every finished instance persists its own CSV row immediately (long sweeps
# survive interruption). Usage:
#   scripts/bench_each.sh <instance_dir> <rows_dir> <out_csv> [extra sundial-bench flags…]
set -uo pipefail
cd "$(dirname "$0")/.."
SRC="$1"; ROWS="$2"; OUT="$3"; shift 3
mkdir -p "$ROWS"
for f in "$SRC"/*.mps; do
  n=$(basename "$f" .mps)
  [ -s "$ROWS/$n.csv" ] && { echo "have    $n"; continue; }
  rm -rf "$ROWS/.single" && mkdir -p "$ROWS/.single"
  cp "$f" "$ROWS/.single/"
  echo "running $n $*..."
  cargo run --release -p sundial-cli -- bench "$ROWS/.single" --out "$ROWS/$n.csv" "$@" 2>&1 | tail -1
done
rm -rf "$ROWS/.single"
head -1 "$(ls "$ROWS"/*.csv | head -1)" > "$OUT"
for c in "$ROWS"/*.csv; do tail -n +2 "$c" >> "$OUT"; done
echo "ASSEMBLED: $(($(wc -l < "$OUT") - 1)) rows in $OUT"
