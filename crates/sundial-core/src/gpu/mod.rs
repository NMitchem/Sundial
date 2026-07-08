pub mod buffers;
pub mod engine; // added in Task 8; create as empty file now
pub mod kernels; // added in Task 7; create as empty file now
pub mod op;

use thiserror::Error;

pub(crate) const WG: u32 = 256;
/// Dispatch cap: kernels grid-stride, so we never exceed WebGPU's
/// guaranteed 65,535 workgroups per dimension (16.7M elems / 256 = 65,536
/// would). 4096 workgroups = 1M threads — plenty of occupancy.
pub(crate) const MAX_WG: u32 = 4096;
pub(crate) fn wgs(len: usize) -> u32 {
    (len as u32).div_ceil(WG).clamp(1, MAX_WG)
}

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter (WebGPU unavailable?)")]
    NoAdapter,
    #[error("device request failed: {0}")]
    Device(String),
    #[error("problem needs a {needed_mib} MiB buffer but adapter allows {allowed_mib} MiB")]
    BufferTooLarge { needed_mib: u64, allowed_mib: u64 },
}

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_name: String,
    pub max_binding_mib: u64,
}

impl GpuContext {
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sundial"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(), // request everything the adapter offers
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;
        Ok(Self {
            device,
            queue,
            adapter_name: format!("{} ({:?})", info.name, info.backend),
            max_binding_mib: limits.max_storage_buffer_binding_size / (1024 * 1024),
        })
    }
}
