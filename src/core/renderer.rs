use image::{ImageBuffer, ImageReader, Rgba};

use crate::config::{Config, Size};

pub struct Renderer {
    width: u32,
    height: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    target_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    render_target: wgpu::Texture,
    texture_bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub async fn new(config: &Config) -> Self {
        let Size { width, height } = config.size;

        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_texture_dimension_2d = 8192;
        limits.max_push_constant_size = 4;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                        | wgpu::Features::PUSH_CONSTANTS,
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .unwrap();

        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../../shaders/shader.wgsl"));

        let target_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&target_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let render_target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let panorama_image = ImageReader::open("textures/panorama.hdr")
            .unwrap()
            .decode()
            .unwrap()
            .into_rgba32f();

        let panorama_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: panorama_image.width(),
                height: panorama_image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &panorama_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            unsafe {
                std::slice::from_raw_parts(
                    panorama_image.as_ptr() as *const u8,
                    panorama_image.len() * 4,
                )
            },
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(panorama_image.width() * 16),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: panorama_image.width(),
                height: panorama_image.height(),
                depth_or_array_layers: 1,
            },
        );

        let panorama_view = panorama_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &texture_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&panorama_view),
            }],
        });

        Self {
            width,
            height,
            device,
            queue,
            target_bind_group_layout,
            pipeline,
            render_target,
            texture_bind_group,
        }
    }

    pub fn render(&self) {
        let view = self
            .render_target
            .create_view(&wgpu::TextureViewDescriptor::default());

        let target_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.target_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });

        for sample in 0..100000 {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            {
                let sample: [u8; 4] = unsafe { std::mem::transmute(sample) };

                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &target_bind_group, &[]);
                compute_pass.set_bind_group(1, &self.texture_bind_group, &[]);
                compute_pass.set_push_constants(0, &sample);
                compute_pass.dispatch_workgroups(self.width / 16, self.height / 16, 1);
            }

            self.queue.submit(Some(encoder.finish()));
        }
    }

    pub async fn retrieve(&self) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.width * self.height * 16) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.render_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.width * 16),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        self.queue.submit(Some(encoder.finish()));

        let (tx, rx) = flume::bounded(1);
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        rx.recv_async().await.unwrap().unwrap();

        let mut buffer = Vec::new();
        {
            let view = slice.get_mapped_range();
            buffer.extend_from_slice(&view[..]);
        }

        staging_buffer.unmap();
        let buffer: Vec<_> = buffer
            .chunks_exact(4)
            .map(TryInto::try_into)
            .map(Result::unwrap)
            .map(f32::from_le_bytes)
            .collect();

        ImageBuffer::from_raw(self.width, self.height, buffer).unwrap()
    }
}
