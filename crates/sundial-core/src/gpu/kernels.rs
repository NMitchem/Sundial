use super::buffers;
use super::GpuContext;
use std::collections::HashMap;

pub struct ParamsData {
    pub n: u32,
    pub stride: u32,
    pub tau: f32,
    pub sigma: f32,
    pub w: f32,
}

impl ParamsData {
    pub fn bytes(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0..4].copy_from_slice(&self.n.to_le_bytes());
        out[4..8].copy_from_slice(&self.stride.to_le_bytes());
        out[8..12].copy_from_slice(&self.tau.to_le_bytes());
        out[12..16].copy_from_slice(&self.sigma.to_le_bytes());
        out[16..20].copy_from_slice(&self.w.to_le_bytes());
        out
    }
}

pub struct Kernels {
    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,
}

const TABLE: &[(&str, &str)] = &[
    ("pdhg", "primal_step"),
    ("pdhg", "dual_step"),
    ("pdhg", "spmv"),
    ("pdhg", "accum"),
    ("pdhg", "ew_scale"),
    ("pdhg", "ew_mul"),
    ("residuals", "primal_res"),
    ("residuals", "dual_res_terms"),
    ("residuals", "row_terms"),
    ("reduce", "reduce_dot"),
    ("reduce", "reduce_sum"),
    ("reduce", "reduce_maxabs"),
    ("reduce", "reduce_diff_sq"),
    ("transport", "ot_apply"),
    ("transport", "ot_apply_t"),
    ("df64", "spmv_df64"),
    ("df64", "ot_apply_df64"),
    ("df64", "reduce_dot_df64"),
    ("df64", "reduce_sum_df64"),
];

impl Kernels {
    pub fn new(device: &wgpu::Device) -> Self {
        let sources: HashMap<&str, &str> = HashMap::from([
            ("pdhg", include_str!("shaders/pdhg.wgsl")),
            ("residuals", include_str!("shaders/residuals.wgsl")),
            ("reduce", include_str!("shaders/reduce.wgsl")),
            ("transport", include_str!("shaders/transport.wgsl")),
            ("df64", include_str!("shaders/df64.wgsl")),
        ]);
        let mut modules: HashMap<&str, wgpu::ShaderModule> = HashMap::new();
        for (name, src) in sources {
            modules.insert(
                name,
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(src.into()),
                }),
            );
        }
        let mut pipelines = HashMap::new();
        for &(module, entry) in TABLE {
            pipelines.insert(
                entry,
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: None, // auto layout: exactly the bindings this entry point references
                    module: &modules[module],
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }
        Self { pipelines }
    }
    pub fn pipeline(&self, name: &str) -> &wgpu::ComputePipeline {
        &self.pipelines[name]
    }
}

pub fn bind(
    device: &wgpu::Device,
    pipeline: &wgpu::ComputePipeline,
    entries: &[(u32, &wgpu::Buffer)],
) -> wgpu::BindGroup {
    let bge: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|&(slot, buf)| wgpu::BindGroupEntry {
            binding: slot,
            resource: buf.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &bge,
    })
}

pub(crate) fn pass_dispatch(
    enc: &mut wgpu::CommandEncoder,
    pl: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    wgs: u32,
) {
    let mut pass = enc.begin_compute_pass(&Default::default());
    pass.set_pipeline(pl);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(wgs.max(1), 1, 1);
}

/// Multi-pass reduction driver. Scratch buffers are allocated once for max_len.
pub struct Reducer {
    scratch_a: wgpu::Buffer,
    scratch_b: wgpu::Buffer,
    df64: bool,
}

impl Reducer {
    pub fn new(device: &wgpu::Device, max_len: usize) -> Self {
        Self::new_with_precision(device, max_len, false)
    }

    pub fn new_with_precision(device: &wgpu::Device, max_len: usize, df64: bool) -> Self {
        let max_partials = max_len.div_ceil(256).clamp(1, 4096);
        Self {
            scratch_a: buffers::storage_zeros_f32(device, max_partials, "reduce_a"),
            scratch_b: buffers::storage_zeros_f32(device, max_partials, "reduce_b"),
            df64,
        }
    }

    /// df64 mode swaps the dot/sum entry points; maxabs is precision-neutral.
    fn entry(&self, name: &'static str) -> &'static str {
        if !self.df64 {
            return name;
        }
        match name {
            "reduce_dot" => "reduce_dot_df64",
            "reduce_sum" => "reduce_sum_df64",
            other => other,
        }
    }

    /// Record a full multi-pass reduction into `enc`; the final scalar is
    /// copied into `results[slot]`. Reductions recorded earlier in the same
    /// encoder complete (submission order) before later ones reuse scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        dev: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        k: &Kernels,
        first: &'static str,
        a: &wgpu::Buffer,
        b: Option<&wgpu::Buffer>,
        n: usize,
        follow: &'static str,
        results: &wgpu::Buffer,
        slot: u32,
    ) {
        // df64 mode remaps dot/sum to their double-double entry points; off, the
        // names pass through unchanged so pipeline selection is byte-identical.
        let first = self.entry(first);
        let follow = self.entry(follow);
        let mut len = n;
        let mut src: &wgpu::Buffer = a;
        let mut dst = &self.scratch_a;
        let mut entry = first;
        loop {
            let wgs = len.div_ceil(256).min(4096) as u32;
            let params = ParamsData {
                n: len as u32,
                stride: wgs * 256,
                tau: 0.0,
                sigma: 0.0,
                w: 0.0,
            };
            let ubuf = buffers::uniform_bytes(dev, &params.bytes(), "reduce_params");
            let pl = k.pipeline(entry);
            let mut e: Vec<(u32, &wgpu::Buffer)> = vec![(0, &ubuf), (1, src), (6, dst)];
            if entry.starts_with("reduce_dot") || entry.starts_with("reduce_diff_sq") {
                e.push((2, b.expect("two-input reduction needs b")));
            }
            let bg = bind(dev, pl, &e);
            pass_dispatch(enc, pl, &bg, wgs);
            len = wgs as usize;
            if len == 1 {
                enc.copy_buffer_to_buffer(dst, 0, results, (slot as u64) * 4, 4);
                return;
            }
            src = dst;
            dst = if std::ptr::eq(dst, &self.scratch_a) {
                &self.scratch_b
            } else {
                &self.scratch_a
            };
            entry = follow;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_dot(
        &self,
        dev: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        k: &Kernels,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n: usize,
        results: &wgpu::Buffer,
        slot: u32,
    ) {
        self.record(
            dev,
            enc,
            k,
            "reduce_dot",
            a,
            Some(b),
            n,
            "reduce_sum",
            results,
            slot,
        );
    }
    /// ‖a − b‖² into `results[slot]` (take the sqrt on the host). Note the
    /// follow-up passes are plain `reduce_sum`: only the FIRST pass differences,
    /// after which it is an ordinary sum of partials.
    #[allow(clippy::too_many_arguments)]
    pub fn record_diff_sq(
        &self,
        dev: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        k: &Kernels,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n: usize,
        results: &wgpu::Buffer,
        slot: u32,
    ) {
        self.record(
            dev,
            enc,
            k,
            "reduce_diff_sq",
            a,
            Some(b),
            n,
            "reduce_sum",
            results,
            slot,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_sum(
        &self,
        dev: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        k: &Kernels,
        a: &wgpu::Buffer,
        n: usize,
        results: &wgpu::Buffer,
        slot: u32,
    ) {
        self.record(
            dev,
            enc,
            k,
            "reduce_sum",
            a,
            None,
            n,
            "reduce_sum",
            results,
            slot,
        );
    }
    #[allow(clippy::too_many_arguments)]
    pub fn record_maxabs(
        &self,
        dev: &wgpu::Device,
        enc: &mut wgpu::CommandEncoder,
        k: &Kernels,
        a: &wgpu::Buffer,
        n: usize,
        results: &wgpu::Buffer,
        slot: u32,
    ) {
        self.record(
            dev,
            enc,
            k,
            "reduce_maxabs",
            a,
            None,
            n,
            "reduce_maxabs",
            results,
            slot,
        );
    }

    #[allow(clippy::too_many_arguments)]
    async fn run(
        &self,
        ctx: &GpuContext,
        k: &Kernels,
        first: &'static str,
        a: &wgpu::Buffer,
        b: Option<&wgpu::Buffer>,
        n: usize,
        follow: &'static str,
    ) -> f32 {
        let results = buffers::storage_zeros_f32(&ctx.device, 1, "reduce_result");
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        self.record(
            &ctx.device,
            &mut enc,
            k,
            first,
            a,
            b,
            n,
            follow,
            &results,
            0,
        );
        ctx.queue.submit([enc.finish()]);
        buffers::readback_f32(&ctx.device, &ctx.queue, &results, 1).await[0]
    }

    pub async fn dot(
        &self,
        ctx: &GpuContext,
        k: &Kernels,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n: usize,
    ) -> f32 {
        self.run(ctx, k, "reduce_dot", a, Some(b), n, "reduce_sum")
            .await
    }
    /// ‖a − b‖² (host takes the sqrt).
    pub async fn diff_sq(
        &self,
        ctx: &GpuContext,
        k: &Kernels,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        n: usize,
    ) -> f32 {
        self.run(ctx, k, "reduce_diff_sq", a, Some(b), n, "reduce_sum")
            .await
    }
    pub async fn sum(&self, ctx: &GpuContext, k: &Kernels, a: &wgpu::Buffer, n: usize) -> f32 {
        self.run(ctx, k, "reduce_sum", a, None, n, "reduce_sum")
            .await
    }
    pub async fn maxabs(&self, ctx: &GpuContext, k: &Kernels, a: &wgpu::Buffer, n: usize) -> f32 {
        self.run(ctx, k, "reduce_maxabs", a, None, n, "reduce_maxabs")
            .await
    }
}
