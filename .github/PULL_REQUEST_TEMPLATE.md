<!--
One logical change per PR. See CONTRIBUTING.md — especially the invariants section,
which lists the changes that will be declined on principle no matter how well written.
-->

## What this changes

<!-- And why. Link the issue if there is one. -->

## Evidence

<!--
Paste the test output that backs the claim. "Tests pass" without the output is not
evidence. If you changed a benchmark number, say what hardware and what tolerance
produced it.
-->

```
```

## Checklist

- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, and `cargo test --workspace` pass locally.
- [ ] New behavior has a test; a bug fix has a test that fails before the fix.
- [ ] **If I touched `crates/sundial-core/src/gpu/` (including WGSL):** I ran `cargo test --workspace -- --include-ignored` on a real GPU and pasted the output above. The 1M transport gate was run in `--release`.
- [ ] **If I changed a documented number anywhere:** `docs/STATUS.md` is updated in this same PR, with its caveats still attached.
- [ ] This does not weaken any invariant in CONTRIBUTING.md — or, if it does, I said which one and why that is correct.
