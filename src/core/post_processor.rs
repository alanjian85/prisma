use std::{collections::HashMap, error::Error, rc::Rc, sync::mpsc};

use image::RgbaImage;

use crate::config::{Config, Size};

use super::RenderContext;

pub struct PostProcessor {
    context: Rc<RenderContext>,
    width: u32,
    height: u32,
    aligned_width: u32,
    aligned_height: u32,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    dst_texture: wgpu::Texture,
}

impl PostProcessor {
    pub fn new(context: Rc<RenderContext>, config: &Config) -> Self {
        let device = context.device();

        let Size { width, height } = config.size;
        let aligned_width = (width + 15) / 16 * 16;
        let aligned_height = (height + 15) / 16 * 16;

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../../shaders/post_process.wgsl"));

        let mut constants = HashMap::new();
        constants.insert(String::from("NUM_SAMPLES"), config.samples as f64);

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: true,
                vertex_pulling_transform: false,
            },
            cache: None,
        });

        let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: aligned_width,
                height: aligned_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        Self {
            context,
            width,
            height,
            aligned_width,
            aligned_height,
            bind_group_layout,
            pipeline,
            dst_texture,
        }
    }

    pub fn post_process(&self, src_texture: &wgpu::Texture) {
        let device = self.context.device();
        let queue = self.context.queue();

        let src_view = src_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = self
            .dst_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(self.aligned_width / 16, self.aligned_height / 16, 1);
        }

        queue.submit(Some(encoder.finish()));
    }

    pub async fn retrieve_result(&self) -> Result<Option<RgbaImage>, Box<dyn Error>> {
        let device = self.context.device();
        let queue = self.context.queue();

        let padded_width = (self.width + 63) / 64 * 64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (padded_width * self.height * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.dst_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_width * 4),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        queue.submit(Some(encoder.finish()));

        let (tx, rx) = mpsc::channel();
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::Maintain::Wait).panic_on_timeout();
        rx.recv()??;

        let mut buffer = Vec::new();
        {
            let view = slice.get_mapped_range();
            buffer.extend_from_slice(&view[..]);
        }
        staging_buffer.unmap();

        let mut image_buffer = Vec::with_capacity((self.width * self.height * 4) as usize);
        for y in 0..self.height {
            for x in 0..self.width {
                for i in 0..4 {
                    image_buffer.push(buffer[((y * padded_width + x) * 4 + i) as usize]);
                }
            }
        }

        Ok(RgbaImage::from_raw(self.width, self.height, image_buffer))
    }
}
