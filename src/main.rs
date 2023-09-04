// Imports --------------------------------------------------------------------
extern crate metal;
extern crate rand;
use {
    metal::*,
    rand::{thread_rng, Rng},
    std::sync::Arc,
};

// Types ----------------------------------------------------------------------
pub struct GPU {
    dev: Arc<Device>,
    cmds: Arc<CommandQueue>,
    state: Arc<ComputePipelineState>,
}

// Public Functions -----------------------------------------------------------
pub fn main() {
    let size = 1_000_000_000;
    let gpu = setup();
    loop {
        let rand_start = std::time::Instant::now();
        let a: Vec<f32> = (0..size)
            .map(|_| thread_rng().gen_range(-1.0..=1.0))
            .collect();
        let b: Vec<f32> = (0..size)
            .map(|_| thread_rng().gen_range(-1.0..=1.0))
            .collect();
        let rand_elapsed = rand_start.elapsed();

        let gpu_start = std::time::Instant::now();
        let result = gpu_dot(&a, &b, &gpu);
        let gpu_elapsed = gpu_start.elapsed();

        let cpu_start = std::time::Instant::now();
        let cpu_result = cpu_dot(&a, &b);
        let cpu_elapsed = cpu_start.elapsed();
        assert_eq!(result, cpu_result);

        println!("Matrix time : {:?}", rand_elapsed);
        println!();
        println!("GPU dot     : {}", result);
        println!("GPU time    : {:?}", gpu_elapsed);
        println!();
        println!("CPU dot     : {}", cpu_result);
        println!("CPU time    : {:?}", cpu_elapsed);
    }
}

pub fn cpu_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut result = 0.0;
    for i in 0..a.len() {
        result += a[i] * b[i];
    }
    result
}

pub fn gpu_dot(a: &[f32], b: &[f32], gpu: &GPU) -> f32 {
    let command_buffer = gpu.cmds.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    let buf_result = set_bufs(Arc::clone(&gpu.dev), encoder, a, b);
    encoder.set_compute_pipeline_state(&gpu.state);
    run_func(command_buffer, encoder, a.len());
    result(buf_result, a.len())
}

pub fn setup() -> GPU {
    let dev = Arc::new(Device::system_default().expect("Apple Metal GPU"));
    let cmds = Arc::new(dev.new_command_queue());
    let state = Arc::new(state(Arc::clone(&dev)).unwrap());
    GPU { dev, cmds, state }
}

pub fn result(buf_result: Buffer, len: usize) -> f32 {
    unsafe { std::slice::from_raw_parts(buf_result.contents() as *const _, len) }
        .iter()
        .sum()
}

pub fn run_func(command_buffer: &CommandBufferRef, encoder: &ComputeCommandEncoderRef, len: usize) {
    let threads_per_threadgroup: u64 = 64;
    let threadgroups = (len as f64 / threads_per_threadgroup as f64).ceil() as u64;
    encoder.dispatch_thread_groups(size_1d(threadgroups), size_1d(threads_per_threadgroup));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}

pub fn set_bufs(dev: Arc<Device>, enc: &ComputeCommandEncoderRef, a: &[f32], b: &[f32]) -> Buffer {
    let buf_result = buf(&dev, &vec![0.0; a.len()]);
    enc.set_buffer(0, Some(&buf(&dev, a)), 0);
    enc.set_buffer(1, Some(&buf(&dev, b)), 0);
    enc.set_buffer(2, Some(&buf_result), 0);
    buf_result
}

pub fn size_1d(width: u64) -> MTLSize {
    let (height, depth) = (1, 1);
    MTLSize {
        width,
        height,
        depth,
    }
}

pub fn buf(dev: &Arc<Device>, buf: &[f32]) -> Buffer {
    dev.new_buffer_with_data(
        buf.as_ptr() as *const std::ffi::c_void,
        (buf.len() * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

pub fn func(device: &Arc<Device>, fn_name: String) -> Result<Function, String> {
    let source = include_str!("shader.metal");
    let options = metal::CompileOptions::new();
    let library = device
        .new_library_with_source(source, options.as_ref())
        .unwrap();
    library.get_function(&fn_name, None)
}

pub fn state(device: Arc<Device>) -> Result<ComputePipelineState, String> {
    let dot_fn = func(&device, "dot_product".to_string()).unwrap();
    let pipeline_desc = ComputePipelineDescriptor::new();
    pipeline_desc.set_compute_function(Some(&dot_fn));
    device.new_compute_pipeline_state(&pipeline_desc)
}
