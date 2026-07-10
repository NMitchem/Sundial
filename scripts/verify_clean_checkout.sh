#!/usr/bin/env bash
# Prove a fresh clone builds everything a new contributor / CI needs, with
# no dependence on untracked local state. Uses a scratch clone of HEAD.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
echo "== clean clone → $SCRATCH"
git clone --quiet "$REPO_ROOT" "$SCRATCH/sundial"
cd "$SCRATCH/sundial"
echo "== rust gates"
cargo fmt --all --check
cargo test --workspace
cargo build -p sundial-lp --target wasm32-unknown-unknown
echo "== wasm-pack + web"
wasm-pack build crates/sundial-web --target web
cd web && npm install && npx tsc --noEmit && npm run build
echo "CLEAN CHECKOUT: ALL GREEN"
