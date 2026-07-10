#!/usr/bin/env bash
# Build the sundial-lp npm package into crates/sundial-web/pkg/ (NOT published).
set -euo pipefail
cd "$(dirname "$0")/.."
wasm-pack build crates/sundial-web --target web --release
cp crates/sundial-web/types-extra.d.ts crates/sundial-web/pkg/
cd crates/sundial-web/pkg
npm pkg set 'files[]=types-extra.d.ts'
# wasm-pack copies LICENSE-* into pkg/ but doesn't add them to the "files"
# allowlist (only bare LICENSE/LICENCE is auto-packed by npm) — list them
# explicitly so the dual-license files actually ship in the tarball.
npm pkg set 'files[]=LICENSE-APACHE' 'files[]=LICENSE-MIT'
echo "== pack dry run"
npm pack --dry-run
