use std::collections::HashMap;
use std::io::Read;
use sundial_core::problem::{CsrMatrix, LpProblem};
use thiserror::Error;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[derive(Debug, Error)]
pub enum MpsError {
    #[error("line {line}: {msg}")]
    Parse { line: usize, msg: String },
    #[error("gzip: {0}")]
    Gzip(String),
    #[error("problem construction: {0}")]
    Problem(String),
}

fn err(line: usize, msg: impl Into<String>) -> MpsError {
    MpsError::Parse {
        line,
        msg: msg.into(),
    }
}

pub fn parse_bytes(bytes: &[u8]) -> Result<LpProblem, MpsError> {
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        let mut s = String::new();
        flate2::read::GzDecoder::new(bytes)
            .read_to_string(&mut s)
            .map_err(|e| MpsError::Gzip(e.to_string()))?;
        parse_str(&s)
    } else {
        parse_str(std::str::from_utf8(bytes).map_err(|e| MpsError::Gzip(e.to_string()))?)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum RowType {
    N,
    L,
    G,
    E,
}

enum Section {
    Start,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Done,
}

pub fn parse_str(text: &str) -> Result<LpProblem, MpsError> {
    let inf = f64::INFINITY;
    let mut name = String::from("unnamed");
    let mut section = Section::Start;

    let mut obj_row: Option<String> = None;
    let mut row_types: Vec<RowType> = Vec::new(); // constraint rows only
    let mut row_index: HashMap<String, usize> = HashMap::new();
    let mut rhs_pre_range: Vec<f64> = Vec::new(); // rhs value per constraint row (default 0)

    let mut col_index: HashMap<String, usize> = HashMap::new();
    let mut col_order: Vec<String> = Vec::new();
    let mut c: Vec<f64> = Vec::new();
    let mut obj_offset = 0.0f64;
    // triplets keyed by (row, col) so duplicates sum
    let mut entries: HashMap<(usize, usize), f64> = HashMap::new();
    let mut ranges: Vec<Option<f64>> = Vec::new(); // per row
    let mut bounds_lo: Vec<Option<f64>> = Vec::new(); // per col, None = default 0
    let mut bounds_up: Vec<Option<f64>> = Vec::new(); // per col, None = default +inf
    let mut up_negative: Vec<bool> = Vec::new(); // per col, true if UP < 0 seen

    for (lineno0, raw) in text.lines().enumerate() {
        let lineno = lineno0 + 1;
        if raw.starts_with('*') || raw.trim().is_empty() {
            continue;
        }
        let is_header = !raw.starts_with(' ') && !raw.starts_with('\t');
        let toks: Vec<&str> = raw.split_whitespace().collect();
        if is_header {
            match toks[0] {
                "NAME" => {
                    if toks.len() > 1 {
                        name = toks[1].to_string();
                    }
                }
                "OBJSENSE" => return Err(err(lineno, "OBJSENSE unsupported in M0 (MIN only)")),
                "ROWS" => section = Section::Rows,
                "COLUMNS" => section = Section::Columns,
                "RHS" => section = Section::Rhs,
                "RANGES" => section = Section::Ranges,
                "BOUNDS" => section = Section::Bounds,
                "ENDATA" => {
                    section = Section::Done;
                    break;
                }
                other => return Err(err(lineno, format!("unknown section '{other}'"))),
            }
            continue;
        }
        match section {
            Section::Rows => {
                if toks.len() < 2 {
                    return Err(err(lineno, "ROWS line needs 'type name'"));
                }
                let ty = match toks[0] {
                    "N" => RowType::N,
                    "L" => RowType::L,
                    "G" => RowType::G,
                    "E" => RowType::E,
                    other => return Err(err(lineno, format!("bad row type '{other}'"))),
                };
                let rname = toks[1].to_string();
                if ty == RowType::N {
                    if obj_row.is_none() {
                        obj_row = Some(rname); // later N rows ignored
                    }
                } else {
                    if row_index.contains_key(&rname) {
                        return Err(err(lineno, format!("duplicate row '{rname}'")));
                    }
                    row_index.insert(rname, row_types.len());
                    row_types.push(ty);
                    rhs_pre_range.push(0.0);
                    ranges.push(None);
                }
            }
            Section::Columns => {
                if toks.len() >= 3 && toks[1].contains("MARKER") {
                    return Err(err(lineno, "integer variables unsupported"));
                }
                if toks.len() < 3 || toks.len().is_multiple_of(2) {
                    return Err(err(lineno, "COLUMNS line needs 'col row val [row val]'"));
                }
                let cname = toks[0];
                let j = *col_index.entry(cname.to_string()).or_insert_with(|| {
                    col_order.push(cname.to_string());
                    c.push(0.0);
                    bounds_lo.push(None);
                    bounds_up.push(None);
                    up_negative.push(false);
                    col_order.len() - 1
                });
                for pair in toks[1..].chunks(2) {
                    let val: f64 = pair[1]
                        .parse()
                        .map_err(|_| err(lineno, format!("bad number '{}'", pair[1])))?;
                    if Some(pair[0]) == obj_row.as_deref() {
                        c[j] += val;
                    } else if let Some(&r) = row_index.get(pair[0]) {
                        *entries.entry((r, j)).or_insert(0.0) += val;
                    } else {
                        return Err(err(lineno, format!("unknown row '{}'", pair[0])));
                    }
                }
            }
            Section::Rhs => {
                if toks.len() < 3 || toks.len().is_multiple_of(2) {
                    return Err(err(lineno, "RHS line needs 'set row val [row val]'"));
                }
                for pair in toks[1..].chunks(2) {
                    let val: f64 = pair[1]
                        .parse()
                        .map_err(|_| err(lineno, format!("bad number '{}'", pair[1])))?;
                    if Some(pair[0]) == obj_row.as_deref() {
                        obj_offset = -val; // MPS convention
                    } else if let Some(&r) = row_index.get(pair[0]) {
                        rhs_pre_range[r] = val;
                    } else {
                        return Err(err(lineno, format!("unknown row '{}'", pair[0])));
                    }
                }
            }
            Section::Ranges => {
                if toks.len() < 3 || toks.len().is_multiple_of(2) {
                    return Err(err(lineno, "RANGES line needs 'set row val [row val]'"));
                }
                for pair in toks[1..].chunks(2) {
                    let val: f64 = pair[1]
                        .parse()
                        .map_err(|_| err(lineno, format!("bad number '{}'", pair[1])))?;
                    let &r = row_index
                        .get(pair[0])
                        .ok_or_else(|| err(lineno, format!("unknown row '{}'", pair[0])))?;
                    ranges[r] = Some(val);
                }
            }
            Section::Bounds => {
                let bt = toks[0];
                let needs_val = matches!(bt, "UP" | "LO" | "FX");
                let min_toks = if needs_val { 4 } else { 3 };
                if toks.len() < min_toks {
                    return Err(err(
                        lineno,
                        format!("BOUNDS '{bt}' needs {min_toks} fields"),
                    ));
                }
                let &j = col_index
                    .get(toks[2])
                    .ok_or_else(|| err(lineno, format!("unknown column '{}'", toks[2])))?;
                let val = if needs_val {
                    toks[3]
                        .parse::<f64>()
                        .map_err(|_| err(lineno, format!("bad number '{}'", toks[3])))?
                } else {
                    0.0
                };
                match bt {
                    "UP" => {
                        bounds_up[j] = Some(val);
                        if val < 0.0 {
                            up_negative[j] = true;
                        }
                    }
                    "LO" => bounds_lo[j] = Some(val),
                    "FX" => {
                        bounds_lo[j] = Some(val);
                        bounds_up[j] = Some(val);
                    }
                    "FR" => {
                        bounds_lo[j] = Some(-inf);
                        bounds_up[j] = Some(inf);
                    }
                    "MI" => bounds_lo[j] = Some(-inf),
                    "PL" => bounds_up[j] = Some(inf),
                    "BV" | "UI" | "LI" => return Err(err(lineno, "integer bounds unsupported")),
                    other => return Err(err(lineno, format!("bad bound type '{other}'"))),
                }
            }
            Section::Start | Section::Done => {
                return Err(err(lineno, "data line outside any section"));
            }
        }
    }
    if !matches!(section, Section::Done) {
        return Err(err(text.lines().count(), "missing ENDATA"));
    }
    if obj_row.is_none() {
        return Err(err(0, "no objective (N) row"));
    }

    // row bounds from type + rhs + ranges
    let m = row_types.len();
    let mut row_lower = vec![-inf; m];
    let mut row_upper = vec![inf; m];
    for r in 0..m {
        let b = rhs_pre_range[r];
        match row_types[r] {
            RowType::L => row_upper[r] = b,
            RowType::G => row_lower[r] = b,
            RowType::E => {
                row_lower[r] = b;
                row_upper[r] = b;
            }
            RowType::N => unreachable!(),
        }
        if let Some(rg) = ranges[r] {
            match row_types[r] {
                RowType::L => row_lower[r] = row_upper[r] - rg.abs(),
                RowType::G => row_upper[r] = row_lower[r] + rg.abs(),
                RowType::E => {
                    if rg >= 0.0 {
                        row_upper[r] = b + rg;
                    } else {
                        row_lower[r] = b + rg;
                    }
                }
                RowType::N => unreachable!(),
            }
        }
    }

    // column bounds from defaults + BOUNDS
    let n = col_order.len();
    // Classical MPS quirk: a negative UP bound on a column with NO explicit
    // lower bound implies lower = -inf rather than the default 0 (otherwise
    // the column would be infeasible-by-default; lp_solve/CPLEX semantics).
    // An explicit LO/FX/MI/FR wins regardless of line order.
    let col_lower: Vec<f64> = (0..n)
        .map(|j| bounds_lo[j].unwrap_or(if up_negative[j] { -inf } else { 0.0 }))
        .collect();
    let col_upper: Vec<f64> = (0..n).map(|j| bounds_up[j].unwrap_or(inf)).collect();

    // triplets -> CSR (sorted by row, then col)
    let mut trips: Vec<(usize, usize, f64)> =
        entries.into_iter().map(|((r, j), v)| (r, j, v)).collect();
    trips.sort_unstable_by_key(|&(r, j, _)| (r, j));
    let mut indptr = vec![0u32; m + 1];
    for &(r, _, _) in &trips {
        indptr[r + 1] += 1;
    }
    for r in 0..m {
        indptr[r + 1] += indptr[r];
    }
    let indices: Vec<u32> = trips.iter().map(|&(_, j, _)| j as u32).collect();
    let values: Vec<f64> = trips.iter().map(|&(_, _, v)| v).collect();
    let a = CsrMatrix {
        n_rows: m,
        n_cols: n,
        indptr,
        indices,
        values,
    };

    LpProblem::new(
        name, a, c, obj_offset, row_lower, row_upper, col_lower, col_upper,
    )
    .map_err(|e| MpsError::Problem(e.to_string()))
}
