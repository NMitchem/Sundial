//! Render `sundial bench` CSV output + known optima into an honest
//! markdown table. Non-Optimal rows are reported as-is — the table's
//! credibility IS the product; never filter them out.
use std::collections::HashMap;

pub fn parse_optima(text: &str) -> HashMap<String, f64> {
    text.lines()
        .skip(1)
        .filter_map(|l| {
            let (name, v) = l.split_once(',')?;
            Some((name.trim().to_string(), v.trim().parse().ok()?))
        })
        .collect()
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

pub fn render(csv: &str, optima: &HashMap<String, f64>) -> String {
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
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f = split_csv_line(line);
        if f.len() < 2 {
            continue;
        }
        total += 1;
        let name = &f[0];
        let status = &f[1];
        let obj: Option<f64> = f.get(2).and_then(|s| s.parse().ok());
        let (known, rel) = match (optima.get(name.as_str()), obj) {
            (Some(&k), Some(o)) => {
                let r = (o - k).abs() / (1.0 + k.abs());
                (format!("{k}"), format!("{r:.1e}"))
            }
            _ => ("—".into(), "—".into()),
        };
        if status == "Optimal" {
            optimal += 1;
            if let (Some(&k), Some(o)) = (optima.get(name.as_str()), obj) {
                let r = (o - k).abs() / (1.0 + k.abs());
                if worst.as_ref().is_none_or(|(_, w)| r > *w) {
                    worst = Some((name.clone(), r));
                }
            }
        }
        let cell = |i: usize| f.get(i).cloned().unwrap_or_default();
        out.push_str(&format!(
            "| {name} | {status} | {} | {known} | {rel} | {} | {} | {} | {} | {} |\n",
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
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const CSV: &str = "name,status,objective,iterations,solve_ms,rel_primal,rel_dual,rel_gap\n\
        \"afiro\",Optimal,-464.7530946,4352,89.2,4.1e-5,3.2e-5,9.8e-5\n\
        \"kb2\",IterationLimit,-1749.9,500000,60000.0,2.1e-3,8.0e-4,4.0e-3\n\
        \"broken\",Error: parsing bench/netlib/broken.mps: bad number 'x',,,,,,\n";

    fn optima() -> HashMap<String, f64> {
        HashMap::from([("afiro".to_string(), -464.75314286)])
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
        let mut o = HashMap::new();
        o.insert("we\"ird".to_string(), 1.0);
        let csv = "name,status,objective,iterations,solve_ms,rel_primal,rel_dual,rel_gap\n\
                   \"we\"\"ird\",Optimal,1.0,1,1.0,1e-5,1e-5,1e-5\n";
        let md = render(csv, &o);
        assert!(md.contains("| we\"ird | Optimal |"), "{md}");
    }

    #[test]
    fn parse_optima_reads_csv() {
        let m = parse_optima("name,objective\nafiro,-464.75314286\n");
        assert_eq!(m["afiro"], -464.75314286);
    }
}
