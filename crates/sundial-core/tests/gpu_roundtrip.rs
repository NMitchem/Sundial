use sundial_core::gpu::{buffers, GpuContext};

#[test]
#[ignore = "requires GPU"]
fn storage_buffer_roundtrip() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU adapter");
    let data: Vec<f32> = (0..1027).map(|i| i as f32 * 0.5).collect();
    let buf = buffers::storage_f32(&ctx.device, &data, "roundtrip");
    let back = pollster::block_on(buffers::readback_f32(
        &ctx.device,
        &ctx.queue,
        &buf,
        data.len(),
    ));
    assert_eq!(back, data);
    assert!(!ctx.adapter_name.is_empty());
}
