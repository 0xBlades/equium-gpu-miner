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
        // Linux/headless order: Vulkan first (best compute support on
        // NVIDIA), then OpenGL compute, then let wgpu auto-pick.
        let backends = [
            wgpu::Backends::VULKAN,
            wgpu::Backends::GL,
            wgpu::Backends::PRIMARY,
        ];

        let mut last_err = anyhow!("no backends to try");

        for backend in &backends {
            match Self::try_new_with_backend(*backend) {
                Ok(gpu) => {
                    log::info!("Using GPU backend: {:?} - {}", gpu.backend, gpu.adapter_name);
                    return Ok(gpu);
                }
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

        // On headless Linux servers with NVIDIA GPUs, HighPerformance
        // sometimes fails because the driver does not expose a power
        // profile. Try HighPerformance first, then fall back to None.
        let adapter = pollster::block_on(async {
            for power in [
                wgpu::PowerPreference::HighPerformance,
                wgpu::PowerPreference::None,
            ] {
                if let Some(a) = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: power,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                {
                    return Some(a);
                }
            }
            None
        })
        .ok_or_else(|| {
            let mut msg = format!("no compatible GPU adapter found for backend {:?}", backend);
            let adapters: Vec<_> = instance.enumerate_adapters(backend).collect();
            if !adapters.is_empty() {
                msg.push_str(". Available adapters on this backend:\n");
                for a in adapters {
                    let info = a.get_info();
                    msg.push_str(&format!("  - {} ({:?})\n", info.name, info.device_type));
                }
            } else {
                msg.push_str(". No adapters enumerated for this backend.");
            }
            #[cfg(target_os = "linux")]
            {
                msg.push_str("\n\nLinux troubleshooting:\n");
                msg.push_str("  1. Run 'vulkaninfo --summary' to verify Vulkan is installed and sees your GPU.\n");
                msg.push_str("  2. Ensure the NVIDIA proprietary driver is installed (not nouveau).\n");
                msg.push_str("  3. Install the Vulkan loader: sudo apt install libvulkan1  (Debian/Ubuntu)\n");
                msg.push_str("                                sudo dnf install vulkan-loader (Fedora)\n");
                msg.push_str("  4. Verify ICD exists: ls /usr/share/vulkan/icd.d/nvidia*.json\n");
            }
            anyhow!("{}", msg)
        })?;

        let info = adapter.get_info();
        let backend = info.backend;
        let adapter_name = format!("{} ({:?})", info.name, info.backend);

        log::info!("Adapter found: {} — requesting device...", adapter_name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("equium-gpu-miner"),
                required_features: wgpu::Features::empty(),
                // Use default desktop limits instead of downlevel (mobile/WebGL)
                // limits. RTX 4090 is a desktop GPU; downlevel limits can be
                // unnecessarily restrictive and have caused pipeline issues
                // on some NVIDIA driver versions.
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| anyhow!("device init failed: {e:?}"))?;

        log::info!("Device + queue created.");

        // NOTE: If this segfaults on NVIDIA Linux, the crash is almost
        // certainly inside libnvidia-glvkspirv.so (NVIDIA's SPIR-V
        // compiler) while translating the large unrolled WGSL kernel.
        // There is no Rust-side fix for a driver segfault, but the
        // following workarounds have helped others:
        //   1. Upgrade to the latest NVIDIA driver (>= 555.x).
        //   2. Set VK_ICD_FILENAMES explicitly:
        //        export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
        //   3. Disable the Optimus layer if present:
        //        export VK_LAYER_DISABLE=VK_LAYER_NV_optimus
        //   4. Force synchronous shader compilation:
        //        export __GL_SYNC_TO_VBLANK=0
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("leaves.wgsl"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders/leaves.wgsl"))),
        });

        log::info!("Shader module created (WGSL -> SPIR-V compiled successfully).");

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

        log::info!("Compute pipeline created successfully.");

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
            log::debug!("Dispatching compute: {} workgroups ({} calls, {} leaves)", workgroups, n_calls, n_leaves);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&storage_buf, 0, &staging, 0, storage_bytes);
        log::debug!("Submitting command buffer...");
        self.queue.submit(Some(encoder.finish()));

        log::debug!("Waiting for GPU readback...");
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .context("readback channel closed")?
            .map_err(|e| anyhow!("staging map failed: {e:?}"))?;

        log::debug!("Copying {} bytes from mapped staging buffer", storage_bytes);
        let data = slice.get_mapped_range();
        out[..storage_bytes as usize].copy_from_slice(&data);
        drop(data);
        staging.unmap();

        log::debug!("GPU leaf generation complete.");
        Ok(())
    }
}

