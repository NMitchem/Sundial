# sundial-mps

A parser for **MPS**, the long-standing standard format for exchanging linear
programs. It reads plain `.mps` and gzipped `.mps.gz`, and produces a
[`sundial_core::problem::LpProblem`](https://docs.rs/sundial-core).

This is a pure parser. No GPU, no wasm, no solver dependencies beyond the
problem types themselves, so it's usable on its own if all you need is to read
MPS files.

```rust
let mps = "NAME example
ROWS
 N obj
 L cap
COLUMNS
 x obj -1.0 cap 1.0
RHS
 r cap 4.0
ENDATA
";

let lp = sundial_mps::parse_str(mps).unwrap();
assert_eq!((lp.n_vars(), lp.n_cons()), (1, 1));
```

Use `parse_bytes` for `.mps.gz`. It sniffs the gzip header and decompresses.

It handles the corners real Netlib files actually contain, including
set-name-less `RHS` lines and repeated `UP` bounds on the same column.
`OBJSENSE` isn't supported: problems are minimized.

Part of [Sundial](https://github.com/NMitchem/Sundial), a GPU
linear-programming solver.

## License

MIT OR Apache-2.0, at your option.
