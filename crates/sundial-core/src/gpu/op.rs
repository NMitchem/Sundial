//! GPU-side linear operators: implementations RECORD their `A·x` / `Aᵀ·y`
//! dispatches into the engine's command encoder. The engine never knows
//! whether a matrix exists.
use super::kernels::{self, pass_dispatch, Kernels, ParamsData};
use super::{buffers, wgs, WG};
use crate::problem::CsrMatrix;

pub trait GpuOp {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
    fn record_apply(
        &self,
        dev: &wgpu::Device,
        k: &Kernels,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
    );
    fn record_apply_t(
        &self,
        dev: &wgpu::Device,
        k: &Kernels,
        enc: &mut wgpu::CommandEncoder,
        y: &wgpu::Buffer,
        out: &wgpu::Buffer,
    );
}

/// Explicit CSR pair, uploaded once. `new` takes the ITERATE-space
/// (scaled, for the explicit path) matrix and its transpose.
pub struct CsrGpuOp {
    m: usize,
    n: usize,
    a_indptr: wgpu::Buffer,
    a_indices: wgpu::Buffer,
    a_vals: wgpu::Buffer,
    at_indptr: wgpu::Buffer,
    at_indices: wgpu::Buffer,
    at_vals: wgpu::Buffer,
    u_m: wgpu::Buffer, // spmv params for m-row dispatch (A)
    u_n: wgpu::Buffer, // spmv params for n-row dispatch (Aᵀ)
}

impl CsrGpuOp {
    pub fn new(dev: &wgpu::Device, a: &CsrMatrix, at: &CsrMatrix) -> Self {
        assert_eq!(a.n_rows, at.n_cols);
        assert_eq!(a.n_cols, at.n_rows);
        let f32v = |v: &[f64]| -> Vec<f32> { v.iter().map(|&x| x as f32).collect() };
        let u = |len: usize, label: &str| {
            buffers::uniform_bytes(
                dev,
                &ParamsData {
                    n: len as u32,
                    stride: wgs(len) * WG,
                    tau: 0.0,
                    sigma: 0.0,
                    w: 0.0,
                }
                .bytes(),
                label,
            )
        };
        Self {
            m: a.n_rows,
            n: a.n_cols,
            a_indptr: buffers::storage_u32(dev, &a.indptr, "a_indptr"),
            a_indices: buffers::storage_u32(dev, &a.indices, "a_indices"),
            a_vals: buffers::storage_f32(dev, &f32v(&a.values), "a_vals"),
            at_indptr: buffers::storage_u32(dev, &at.indptr, "at_indptr"),
            at_indices: buffers::storage_u32(dev, &at.indices, "at_indices"),
            at_vals: buffers::storage_f32(dev, &f32v(&at.values), "at_vals"),
            u_m: u(a.n_rows, "csr_u_m"),
            u_n: u(a.n_cols, "csr_u_n"),
        }
    }
}

impl GpuOp for CsrGpuOp {
    fn n_rows(&self) -> usize {
        self.m
    }
    fn n_cols(&self) -> usize {
        self.n
    }
    fn record_apply(
        &self,
        dev: &wgpu::Device,
        k: &Kernels,
        enc: &mut wgpu::CommandEncoder,
        x: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) {
        let pl = k.pipeline("spmv");
        let bg = kernels::bind(
            dev,
            pl,
            &[
                (0, &self.u_m),
                (8, &self.a_indptr),
                (9, &self.a_indices),
                (1, &self.a_vals),
                (2, x),
                (6, out),
            ],
        );
        pass_dispatch(enc, pl, &bg, wgs(self.m));
    }
    fn record_apply_t(
        &self,
        dev: &wgpu::Device,
        k: &Kernels,
        enc: &mut wgpu::CommandEncoder,
        y: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) {
        let pl = k.pipeline("spmv");
        let bg = kernels::bind(
            dev,
            pl,
            &[
                (0, &self.u_n),
                (8, &self.at_indptr),
                (9, &self.at_indices),
                (1, &self.at_vals),
                (2, y),
                (6, out),
            ],
        );
        pass_dispatch(enc, pl, &bg, wgs(self.n));
    }
}
