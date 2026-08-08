//! Render `sundial bench` CSV output + known optima into an honest
//! markdown table. Non-Optimal rows are reported as-is — the table's
//! credibility IS the product; never filter them out.
use std::collections::HashMap;

pub struct Optima {
    pub values: HashMap<String, f64>,
    pub notes: HashMap<String, String>,
}

/// CSV: `name,objective[,note]` — the note column is optional both per row
/// and per file, so older files without a note header still parse.
pub fn parse_optima(text: &str) -> Optima {
    let mut values = HashMap::new();
    let mut notes = HashMap::new();
    for l in text.lines().skip(1) {
        let mut parts = l.splitn(3, ',');
        let (Some(name), Some(v)) = (parts.next(), parts.next()) else {
            continue;
        };
        let Ok(v) = v.trim().parse::<f64>() else {
            continue;
        };
        values.insert(name.trim().to_string(), v);
        if let Some(note) = parts.next() {
            let note = note.trim();
            if !note.is_empty() {
                notes.insert(name.trim().to_string(), note.to_string());
            }
        }
    }
    Optima { values, notes }
}

/// Markdown-table cell hygiene: a literal `|` in error text would split the
/// row; replace with '/'.
fn scrub_cell(s: &str) -> String {
    s.replace('|', "/")
}

/// Minimal CSV field splitter: handles an optionally double-quoted FIRST
/// field with "" escapes (bench quotes only the name field); remaining
/// fields are plain (bench scrubs commas out of them). Malformed input
/// (unterminated quote) degrades to "rest of line is the name" — never
/// panics on hand-edited CSVs.
fn split_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let rest: &str = if let Some(stripped) = line.strip_prefix('"') {
        let mut name = String::new();
        let mut chars = stripped.char_indices().peekable();
        let mut after = ""; // text following the closing quote
        while let Some((i, ch)) = chars.next() {
            if ch == '"' {
                if matches!(chars.peek(), Some((_, '"'))) {
                    name.push('"');
                    chars.next();
                } else {
                    after = &stripped[i + 1..];
                    break;
                }
            } else {
                name.push(ch);
            }
        }
        fields.push(name);
        after.strip_prefix(',').unwrap_or(after)
    } else {
        match line.split_once(',') {
            Some((first, rest)) => {
                fields.push(first.to_string());
                rest
            }
            None => {
                fields.push(line.to_string());
                ""
            }
        }
    };
    if !rest.is_empty() {
        fields.extend(rest.split(',').map(|s| s.to_string()));
    }
    fields
}

const FOOTNOTE_MARKERS: [&str; 5] = ["¹", "²", "³", "⁴", "⁵"];

pub fn render(csv: &str, optima: &Optima) -> String {
    let mut out = String::from(
        "# Sundial benchmark report\n\n\
         GPU results at relative KKT ≤ 1e-4, CPU-f64-verified. Known optima from the \
         Netlib readme; published solver times for the same instances: \
         <https://plato.asu.edu/bench.html>.\n\n\
         | instance | status | objective | known | rel err | iters | ms | primal | dual | gap |\n\
         |---|---|---|---|---|---|---|---|---|---|\n",
    );
    let (mut total, mut optimal) = (0u32, 0u32);
    let mut worst: Option<(String, f64)> = None;
    let mut footnotes: Vec<(String, String)> = Vec::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f = split_csv_line(line);
        if f.len() < 2 {
            continue;
        }
        total += 1;
        let name = &f[0];
        let status = &f[1];
        let obj: Option<f64> = f.get(2).and_then(|s| s.parse().ok());
        let (known, rel) = match (optima.values.get(name.as_str()), obj) {
            (Some(&k), Some(o)) => {
                let r = (o - k).abs() / (1.0 + k.abs());
                (format!("{k}"), format!("{r:.1e}"))
            }
            _ => ("—".into(), "—".into()),
        };
        if status == "Optimal" {
            optimal += 1;
            if let (Some(&k), Some(o)) = (optima.values.get(name.as_str()), obj) {
                let r = (o - k).abs() / (1.0 + k.abs());
                if worst.as_ref().is_none_or(|(_, w)| r > *w) {
                    worst = Some((name.clone(), r));
                }
            }
        }
        let name_cell = match optima.notes.get(name.as_str()) {
            Some(note) => {
                let marker = FOOTNOTE_MARKERS
                    .get(footnotes.len())
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| format!("[{}]", footnotes.len() + 1));
                footnotes.push((marker.clone(), note.clone()));
                format!("{}{marker}", scrub_cell(name))
            }
            None => scrub_cell(name),
        };
        let cell = |i: usize| f.get(i).map(|s| scrub_cell(s)).unwrap_or_default();
        out.push_str(&format!(
            "| {name_cell} | {} | {} | {known} | {rel} | {} | {} | {} | {} | {} |\n",
            scrub_cell(status),
            cell(2),
            cell(3),
            cell(4),
            cell(5),
            cell(6),
            cell(7)
        ));
    }
    out.push_str(&format!(
        "\n**{optimal}/{total} Optimal** (CPU-f64-verified)."
    ));
    if let Some((name, w)) = worst {
        out.push_str(&format!(
            " Worst relative objective error among Optimal: {w:.1e} ({name})."
        ));
    }
    out.push('\n');
    for (marker, note) in &footnotes {
        out.push_str(&format!("\n{marker} {note}\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = "name,status,objective,iterations,solve_ms,rel_primal,rel_dual,rel_gap\n\
        \"afiro\",Optimal,-464.7530946,4352,89.2,4.1e-5,3.2e-5,9.8e-5\n\
        \"kb2\",IterationLimit,-1749.9,500000,60000.0,2.1e-3,8.0e-4,4.0e-3\n\
        \"broken\",Error: parsing bench/netlib/broken.mps: bad number 'x',,,,,,\n";

    fn optima() -> Optima {
        parse_optima("name,objective,note\nafiro,-464.75314286,\n")
    }

    #[test]
    fn renders_status_and_rel_err() {
        let md = render(CSV, &optima());
        assert!(md.contains("| afiro | Optimal |"), "afiro row:\n{md}");
        // |−464.7530946 − (−464.75314286)| / (1 + 464.75314286) = 1.036e-7 → "{:.1e}" = "1.0e-7"
        assert!(md.contains("1.0e-7"), "rel err vs known optimum:\n{md}");
        assert!(
            md.contains("| kb2 | IterationLimit |"),
            "non-optimal rows stay in the table:\n{md}"
        );
        assert!(md.contains("1/3 Optimal"), "summary line:\n{md}");
    }

    #[test]
    fn unknown_instances_get_dash() {
        let md = render(CSV, &optima());
        // kb2 absent from optima → em-dash in known/rel-err columns
        assert!(
            md.lines()
                .any(|l| l.starts_with("| kb2 |") && l.contains("| — |")),
            "{md}"
        );
    }

    #[test]
    fn quoted_names_unescape() {
        let o = parse_optima("name,objective,note\nwe\"ird,1.0,\n");
        let csv = "name,status,objective,iterations,solve_ms,rel_primal,rel_dual,rel_gap\n\
                   \"we\"\"ird\",Optimal,1.0,1,1.0,1e-5,1e-5,1e-5\n";
        let md = render(csv, &o);
        assert!(md.contains("| we\"ird | Optimal |"), "{md}");
    }

    #[test]
    fn parse_optima_reads_csv() {
        let o = parse_optima("name,objective\nafiro,-464.75314286\n");
        assert_eq!(o.values["afiro"], -464.75314286);
    }

    #[test]
    fn note_column_renders_footnote() {
        let o = parse_optima(
            "name,objective,note\ne226,-18.751929,readme value uses the opposite objective-constant sign convention (delta = 2x the RHS constant); our verified optimum is -11.635074\n",
        );
        assert_eq!(o.values["e226"], -18.751929);
        let csv = "name,status,objective,iterations,solve_ms,rel_primal,rel_dual,rel_gap\n\
                   \"e226\",Optimal,-11.635074,52000,12586.0,4.1e-5,3.2e-5,9.8e-5\n";
        let md = render(csv, &o);
        assert!(
            md.contains("e226\u{00b9}") || md.contains("e226 ¹") || md.contains("| e226¹ |"),
            "noted instance gets a superscript marker:\n{md}"
        );
        assert!(
            md.contains("opposite objective-constant sign convention"),
            "footnote text rendered:\n{md}"
        );
    }

    #[test]
    fn two_column_optima_still_parse() {
        let o = parse_optima("name,objective\nafiro,-464.75314286\n");
        assert_eq!(o.values["afiro"], -464.75314286);
        assert!(o.notes.is_empty());
    }

    #[test]
    fn pipes_scrubbed_from_table_cells() {
        let o = optima();
        let csv = "name,status,objective,iterations,solve_ms,rel_primal,rel_dual,rel_gap\n\
                   \"bad\",Error: weird | pipe; more,,,,,,\n";
        let md = render(csv, &o);
        let table_lines: Vec<&str> = md.lines().filter(|l| l.starts_with("| bad")).collect();
        assert_eq!(table_lines.len(), 1);
        assert!(
            !table_lines[0].contains("weird | pipe"),
            "raw pipe must not split the cell:\n{md}"
        );
        assert!(
            table_lines[0].contains("weird / pipe"),
            "pipe replaced with '/':\n{md}"
        );
    }
}
