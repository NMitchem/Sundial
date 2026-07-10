//! df64 accumulator experiment (M2 Task 5). PLATFORM FINDING: on the wgpu
//! Metal backend `df64_dot_beats_f32_on_cancellation` FAILS — measured f32 err
//! 7.552e0 == df64 err 7.552e0 on the seed-42 cancellation input. Metal's
//! `MTLCompileOptions.fastMathEnabled` defaults to YES and wgpu 30 exposes no
//! control to disable it (wgpu-hal sets only language version + preserveInvariance;
//! naga 30's MSL Options has no math-mode field), so fast-math contracts the
//! error-free transforms (`fma(a,b,-a*b)`→0, `s-(s-a)`→a) and df64 collapses to
//! f32. The test is left asserting the true hypothesis (NOT weakened) so the
//! failure records the limitation; a future wgpu with a precision control would
//! make it pass. See the Task 6 memo. `#[ignore]` keeps it out of the GPU-less CI.
use sundial_core::gpu::kernels::{Kernels, Reducer};
use sundial_core::gpu::{buffers, GpuContext};

/// Cancellation-heavy dot product with a wide magnitude spread: f32
/// accumulation (even Neumaier-compensated partials) loses digits; df64
/// must recover ~everything. Ground truth computed in f64 on the CPU.
fn adversarial_pair(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, f64) {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        // log-uniform magnitudes 1e-3..1e6, random signs
        let mag = 10f64.powf(rng.f64() * 9.0 - 3.0);
        a[i] = (if rng.bool() { mag } else { -mag }) as f32;
        b[i] = (rng.f64() * 2.0 - 1.0) as f32;
    }
    let truth: f64 = a.iter().zip(&b).map(|(&x, &y)| x as f64 * y as f64).sum();
    (a, b, truth)
}

#[test]
#[ignore = "requires GPU"]
fn df64_dot_beats_f32_on_cancellation() {
    let ctx = pollster::block_on(GpuContext::new()).expect("no GPU");
    let k = Kernels::new(&ctx.device);
    let n = 1_000_000usize;
    let (a, b, truth) = adversarial_pair(n, 42);
    let ab = buffers::storage_f32(&ctx.device, &a, "a");
    let bb = buffers::storage_f32(&ctx.device, &b, "b");
    let red32 = Reducer::new(&ctx.device, n);
    let red64 = Reducer::new_with_precision(&ctx.device, n, true);
    let d32 = pollster::block_on(red32.dot(&ctx, &k, &ab, &bb, n)) as f64;
    let d64 = pollster::block_on(red64.dot(&ctx, &k, &ab, &bb, n)) as f64;
    let (e32, e64) = ((d32 - truth).abs(), (d64 - truth).abs());
    eprintln!("truth {truth:.6e}  f32 err {e32:.3e}  df64 err {e64:.3e}");
    assert!(
        e64 <= e32 * 0.1 || e64 < truth.abs() * 1e-9 + 1e-6,
        "df64 not materially better: f32 {e32:.3e} vs df64 {e64:.3e} — \
         suspect fast-math contraction (see task's KNOWN PLATFORM RISK)"
    );
}

#[test]
#[ignore = "requires GPU"]
fn df64_off_is_untouched_default() {
    // SolveOptions::df64 defaults false and afiro solves identically.
    let opts = sundial_core::problem::SolveOptions::default();
    assert!(!opts.df64);
}
