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

fn parse_err(text: &str) -> String {
    sundial_mps::parse_str(text).unwrap_err().to_string()
}

#[test]
fn missing_endata_is_an_error() {
    let e = parse_err("NAME t\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\n");
    assert!(e.contains("missing ENDATA"), "{e}");
}

#[test]
fn missing_objective_row_is_an_error() {
    let e = parse_err("NAME t\nROWS\n L r1\nCOLUMNS\n x r1 1.0\nRHS\nENDATA\n");
    assert!(e.contains("no objective"), "{e}");
}

#[test]
fn duplicate_constraint_row_is_an_error() {
    let e =
        parse_err("NAME t\nROWS\n N obj\n L r1\n G r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\nENDATA\n");
    assert!(e.contains("duplicate row 'r1'"), "{e}");
}

#[test]
fn data_line_outside_section_is_an_error() {
    let e = parse_err(" x obj 1.0\nROWS\n N obj\nENDATA\n");
    assert!(e.contains("outside any section"), "{e}");
}

#[test]
fn negative_up_without_lo_opens_lower_bound() {
    // Classical MPS convention: UP < 0 with no explicit LO ⇒ lower = -inf.
    let p = sundial_mps::parse_str(
        "NAME t\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\n rhs r1 5.0\nBOUNDS\n UP bnd x -2.0\nENDATA\n",
    ).unwrap();
    assert_eq!(p.col_upper[0], -2.0);
    assert_eq!(p.col_lower[0], f64::NEG_INFINITY);
}

#[test]
fn negative_up_with_explicit_lo_keeps_lo() {
    let p = sundial_mps::parse_str(
        "NAME t\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\n rhs r1 5.0\nBOUNDS\n LO bnd x -9.0\n UP bnd x -2.0\nENDATA\n",
    ).unwrap();
    assert_eq!(p.col_lower[0], -9.0);
    // order must not matter:
    let p2 = sundial_mps::parse_str(
        "NAME t\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\n rhs r1 5.0\nBOUNDS\n UP bnd x -2.0\n LO bnd x -9.0\nENDATA\n",
    ).unwrap();
    assert_eq!(p2.col_lower[0], -9.0);
}

#[test]
fn rhs_without_set_name_parses() {
    // blend.mps style: RHS lines are bare `row val [row val]` pairs (even
    // token count, no set-name field). Classical MPS permits omitting it.
    let p = sundial_mps::parse_str(
        "NAME t\nROWS\n N obj\n L r1\n G r2\nCOLUMNS\n x obj 1.0 r1 1.0\n x r2 1.0\nRHS\n r1 5.0 r2 1.0\nENDATA\n",
    )
    .unwrap();
    assert_eq!(p.row_upper[0], 5.0); // L row: rhs sets upper
    assert_eq!(p.row_lower[1], 1.0); // G row: rhs sets lower
}

#[test]
fn rhs_with_set_name_still_parses() {
    // regression pin: the odd-token (named-set) form keeps working
    let p = sundial_mps::parse_str(
        "NAME t\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\n rhs r1 5.0\nENDATA\n",
    )
    .unwrap();
    assert_eq!(p.row_upper[0], 5.0);
}

#[test]
fn rhs_set_name_less_unknown_row_is_an_error() {
    let e = sundial_mps::parse_str(
        "NAME t\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\n nope 5.0\nENDATA\n",
    )
    .unwrap_err()
    .to_string();
    // 2 tokens, even ⇒ parsed as a bare `row val` pair; 'nope' isn't a row
    assert!(e.contains("unknown row 'nope'"), "{e}");
}

#[test]
fn repeated_up_lines_reset_negativity() {
    // M1-review hardening: UP -2 then UP +5 must NOT leave lower at -inf
    let p = sundial_mps::parse_str(
        "NAME t\nROWS\n N obj\n L r1\nCOLUMNS\n x obj 1.0 r1 1.0\nRHS\n rhs r1 5.0\nBOUNDS\n UP bnd x -2.0\n UP bnd x 5.0\nENDATA\n",
    )
    .unwrap();
    assert_eq!(p.col_upper[0], 5.0);
    assert_eq!(
        p.col_lower[0], 0.0,
        "negativity flag must reset on the later positive UP"
    );
}
