use sundial_mps::{parse_bytes, parse_str};

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

#[test]
fn parses_afiro_dimensions() {
    let p = parse_bytes(&fixture("afiro.mps")).unwrap();
    assert_eq!(p.n_cons(), 27); // 28 readme rows minus objective
    assert_eq!(p.n_vars(), 32);
    assert!(p.a.nnz() > 80);
}

#[test]
fn parses_all_netlib_fixtures() {
    for f in [
        "afiro.mps",
        "sc50a.mps",
        "sc50b.mps",
        "adlittle.mps",
        "share2b.mps",
    ] {
        let p = parse_bytes(&fixture(f)).unwrap();
        assert!(p.n_vars() > 0 && p.n_cons() > 0, "{f} parsed empty");
    }
}

#[test]
fn ranges_semantics() {
    let p = parse_bytes(&fixture("corner_ranges.mps")).unwrap();
    let idx = |_r: &str| -> usize {
        // rows are in ROWS-section order: R1=0, R2=1, R3=2, R4=3
        match _r {
            "R1" => 0,
            "R2" => 1,
            "R3" => 2,
            "R4" => 3,
            _ => unreachable!(),
        }
    };
    assert_eq!(
        (p.row_lower[idx("R1")], p.row_upper[idx("R1")]),
        (6.0, 10.0)
    ); // L: [u-|R|, u]
    assert_eq!((p.row_lower[idx("R2")], p.row_upper[idx("R2")]), (2.0, 5.0)); // G: [l, l+|R|]
    assert_eq!((p.row_lower[idx("R3")], p.row_upper[idx("R3")]), (5.0, 7.0)); // E, R>0
    assert_eq!((p.row_lower[idx("R4")], p.row_upper[idx("R4")]), (3.0, 5.0)); // E, R<0
}

#[test]
fn bounds_semantics_and_obj_offset() {
    let p = parse_bytes(&fixture("corner_bounds.mps")).unwrap();
    let inf = f64::INFINITY;
    assert_eq!((p.col_lower[0], p.col_upper[0]), (0.0, 4.0)); // UP
    assert_eq!((p.col_lower[1], p.col_upper[1]), (-3.0, inf)); // LO
    assert_eq!((p.col_lower[2], p.col_upper[2]), (2.0, 2.0)); // FX
    assert_eq!((p.col_lower[3], p.col_upper[3]), (-inf, inf)); // FR
    assert_eq!((p.col_lower[4], p.col_upper[4]), (-inf, inf)); // MI (upper stays +inf default)
    assert_eq!(p.obj_offset, 7.5); // RHS on N row => offset = -val
}

#[test]
fn rejects_integer_markers() {
    let text =
        "NAME T\nROWS\n N COST\n L R1\nCOLUMNS\n M 'MARKER' 'INTORG'\n X1 COST 1.0\nENDATA\n";
    assert!(parse_str(text).is_err());
}

#[test]
fn error_carries_line_number() {
    let text = "NAME T\nROWS\n N COST\n Z BADTYPE\nENDATA\n";
    let err = parse_str(text).unwrap_err();
    assert!(format!("{err}").contains("line 4"), "got: {err}");
}
