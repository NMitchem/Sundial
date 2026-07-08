use sundial_core::problem::{CsrMatrix, LpProblem, ProblemError};

fn tiny_csr() -> CsrMatrix {
    CsrMatrix {
        n_rows: 1,
        n_cols: 1,
        indptr: vec![0, 1],
        indices: vec![0],
        values: vec![1.0],
    }
}

#[test]
fn dimension_error_reports_context() {
    let e = LpProblem::new(
        "t".into(),
        tiny_csr(),
        vec![0.0; 5],
        0.0,
        vec![0.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
    )
    .unwrap_err();
    match &e {
        ProblemError::Dimension(msg) => assert!(msg.contains("n=1"), "{msg}"),
        other => panic!("expected Dimension, got {other:?}"),
    }
}

#[test]
fn bound_order_error_reports_indices() {
    let e = LpProblem::new(
        "t".into(),
        tiny_csr(),
        vec![0.0],
        0.0,
        vec![2.0],
        vec![1.0],
        vec![0.0],
        vec![1.0],
    )
    .unwrap_err();
    match &e {
        ProblemError::BoundOrder { kind, index, l, u } => {
            assert_eq!(*kind, "row");
            assert_eq!(*index, 0);
            assert_eq!((*l, *u), (2.0, 1.0));
        }
        other => panic!("expected BoundOrder, got {other:?}"),
    }
    assert!(e.to_string().contains("row index 0"), "{e}");
}
