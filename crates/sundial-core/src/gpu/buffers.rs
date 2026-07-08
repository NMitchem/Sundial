use wgpu::util::DeviceExt;

/// f64 → f32 with ±∞ mapped to the GPU sentinel ±1e30 (see plan's infinity convention).
pub fn pack_f32_inf_sentinel(v: &[f64]) -> Vec<f32> {
    v.iter()
        .map(|&x| {
            if x == f64::INFINITY {
                1e30
            } else if x == f64::NEG_INFINITY {
                -1e30
            } else {
                x as f32
            }
        })
        .collect()
}

pub fn storage_f32(device: &wgpu::Device, data: &[f32], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn storage_u32(device: &wgpu::Device, data: &[u32], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn storage_zeros_f32(device: &wgpu::Device, len: usize, label: &str) -> wgpu::Buffer {
    storage_f32(device, &vec![0.0f32; len.max(1)], label)
}

pub fn uniform_bytes(device: &wgpu::Device, data: &[u8], label: &str) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub async fn readback_f32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    count: usize,
) -> Vec<f32> {
    let size = (count * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit([enc.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    // Native needs an explicit poll to drive the callback; on wasm the browser drives it.
    #[cfg(not(target_arch = "wasm32"))]
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device poll failed");
    rx.await.expect("map_async dropped").expect("map failed");

    let view = slice.get_mapped_range().expect("get_mapped_range failed");
    let out: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    out
}
