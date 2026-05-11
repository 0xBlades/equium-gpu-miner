//! wgpu plumbing for the Equihash 96,5 leaf-generation kernel.
//!
//! Public surface is intentionally narrow:
//!   * `GpuLeafGen::new()` — initialize an adapter + device + pipeline.
//!     Returns `Err` cleanly on systems with no compatible GPU.
//!   * `GpuLeafGen::generate()` — given an I-block + nonce, dispatch
//!     enough invocations to fill an `&mut [u8]` leaves buffer
//!     (size = n_leaves × 12).

use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

/// Equihash 96,5 leaf size in bytes (n / 8).
pub const LEAF_BYTES: usize = 12;

/// 60-byte BLAKE2b output → 5 leaves per call.
pub const LEAVES_PER_CALL: u32 = 5;

/// Workgroup size declared in the WGSL kernel. Used to compute the
/// dispatch grid.
const WORKGROUP_SIZE: u32 = 64;

/// Uniform block matching the shader's `Params` struct.
///
/// WGSL std140 uniform layout requires 16-byte alignment for vec4s and
/// arrays. Every member here is sized + aligned to 16 bytes so the
/// Rust and WGSL layouts agree exactly. Total size: 160 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct Params {
    /// Personalization: [u32_le("Zcas"), u32_le("hPoW"), u32_le(n), u32_le(k)]
    personal: [u32; 4],
    /// [digest_len, n_leaves, padding, padding]
    cfg: [u32; 4],
    /// I-block (81 bytes padded to 96).
    input: [[u32; 4]; 6],
    /// Nonce bytes 0..16.
    nonce: [u32; 4],
    /// Nonce bytes 16..32.
    nonce_hi: [u32; 4],
}

pub struct GpuLeafGen {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub backend: wgpu::Backend,
    pub adapter_name: String,
}

impl GpuLeafGen {
    pub fn new() -> Result<Self> {
        // Try Vulkan first, then fall back to GL (OpenGL compute).
        // The Vulkan backend uses naga → SPIR-V → driver compilation which
        // can crash on some NVIDIA driver versions. The GL backend uses a
        // different shader compilation path that avoids the SPIR-V compiler.
        let backends = [
            wgpu::Backends::VULKAN,
            wgpu::Backends::GL,
        ];

        let mut last_err = anyhow!("no backends to try");

        for backend in &backends {
            match Self::try_new_with_backend(*backend) {
                Ok(gpu) => return Ok(gpu),
                Err(e) => {
                    eprintln!("Backend {:?} failed: {e:#}", backend);
                    last_err = e;
                }
            }
        }
        Err(last_err)
    }

    fn try_new_with_backend(backend: wgpu::Backends) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: backend,
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| anyhow!("no compatible GPU adapter found"))?;

        let info = adapter.get_info();
        let backend = info.backend;
        let adapter_name = format!("{} ({:?})", info.name, info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("equium-gpu-miner"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| anyhow!("device init failed: {e:?}"))?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("leaves.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/leaves.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("leaves.bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("leaves.pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("leaves.pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            backend,
            adapter_name,
        })
    }

    /// Generate `n_leaves` leaves into `out`. `out.len()` must be at
    /// least `n_leaves * LEAF_BYTES`. The CPU host is responsible for
    /// passing a consistent (input, nonce) pair.
    pub fn generate(&self, input: &[u8; 81], nonce: &[u8; 32], n_leaves: u32, out: &mut [u8]) -> Result<()> {
        assert!(out.len() >= n_leaves as usize * LEAF_BYTES);

        // Personalization: "ZcashPoW" + n_le(96) + k_le(5), packed into
        // a single vec4<u32>.
        let mut p = [0u8; 16];
        p[..8].copy_from_slice(b"ZcashPoW");
        p[8..12].copy_from_slice(&96u32.to_le_bytes());
        p[12..16].copy_from_slice(&5u32.to_le_bytes());
        let personal = [
            u32::from_le_bytes(p[0..4].try_into().unwrap()),
            u32::from_le_bytes(p[4..8].try_into().unwrap()),
            u32::from_le_bytes(p[8..12].try_into().unwrap()),
            u32::from_le_bytes(p[12..16].try_into().unwrap()),
        ];

        // Pack the 81-byte input into 6 × vec4<u32> = 24 u32 = 96 bytes.
        // Last 15 bytes are zero; only the first 81 are read by the shader.
        let mut input_padded = [0u8; 96];
        input_padded[..81].copy_from_slice(input);
        let mut input_vec = [[0u32; 4]; 6];
        for i in 0..6 {
            let mut chunk = [0u8; 16];
            chunk.copy_from_slice(&input_padded[i * 16..(i + 1) * 16]);
            input_vec[i] = [
                u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                u32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                u32::from_le_bytes(chunk[12..16].try_into().unwrap()),
            ];
        }

        let mut nonce_lo = [0u32; 4];
        let mut nonce_hi = [0u32; 4];
        for i in 0..4 {
            nonce_lo[i] = u32::from_le_bytes(nonce[i * 4..(i + 1) * 4].try_into().unwrap());
        }
        for i in 0..4 {
            nonce_hi[i] =
                u32::from_le_bytes(nonce[16 + i * 4..16 + (i + 1) * 4].try_into().unwrap());
        }

        let params = Params {
            personal,
            cfg: [60, n_leaves, 0, 0],
            input: input_vec,
            nonce: nonce_lo,
            nonce_hi,
        };

        let uniform_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Storage buffer: 3 × u32 per leaf.
        let storage_bytes = (n_leaves as u64) * (LEAF_BYTES as u64);
        let storage_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("leaves"),
            size: storage_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: storage_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("leaves.bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: storage_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("leaves.enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("leaves.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            // One invocation per BLAKE2b call (= 5 leaves).
            let n_calls = (n_leaves + LEAVES_PER_CALL - 1) / LEAVES_PER_CALL;
            let workgroups = (n_calls + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage_buf, 0, &staging, 0, storage_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .context("readback channel closed")?
            .map_err(|e| anyhow!("staging map failed: {e:?}"))?;

        let data = slice.get_mapped_range();
        out[..storage_bytes as usize].copy_from_slice(&data);
        drop(data);
        staging.unmap();

        Ok(())
    }
}

