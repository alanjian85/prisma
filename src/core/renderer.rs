use std::collections::HashMap;

use image::{ImageBuffer, Rgba};

use crate::config::{Config, Size};

use super::RenderContext;

pub struct Renderer<'a> {
    context: &'a RenderContext,
    width: u32,
    height: u32,
    samples: u32,
    target_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    render_target: wgpu::Texture,
}

pub struct BindGroupLayoutSet<'a> {
    pub texture: &'a wgpu::BindGroupLayout,
}

pub struct BindGroupSet<'a> {
    pub texture: &'a wgpu::BindGroup,
}

impl<'a> Renderer<'a> {
    pub fn new(
        context: &'a RenderContext,
        config: &Config,
        bind_group_layout_set: BindGroupLayoutSet,
    ) -> Self {
        let device = context.device();

        let Size { width, height } = config.size;
        let samples = config.samples;

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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&target_bind_group_layout, bind_group_layout_set.texture],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });

        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../../shaders/shader.wgsl"));

        let mut constants = HashMap::new();
        constants.insert(String::from("MAX_DEPTH"), config.depth as f64);
        constants.insert(String::from("NUM_SAMPLES"), samples as f64);

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

        Self {
            context,
            width,
            height,
            samples,
            target_bind_group_layout,
            pipeline,
            render_target,
        }
    }

    pub fn render(&self, bind_group_set: BindGroupSet) {
        let device = self.context.device();
        let queue = self.context.queue();

        let view = self
            .render_target
            .create_view(&wgpu::TextureViewDescriptor::default());

        let output_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.target_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });

        for sample in 0..self.samples {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            {
                let sample: [u8; 4] = unsafe { std::mem::transmute(sample) };

                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &output_bind_group, &[]);
                compute_pass.set_bind_group(1, bind_group_set.texture, &[]);
                compute_pass.set_push_constants(0, &sample);
                compute_pass.dispatch_workgroups(self.width / 16, self.height / 16, 1);
            }

            queue.submit(Some(encoder.finish()));
        }
    }

    pub async fn retrieve_result(&self) -> ImageBuffer<Rgba<f32>, Vec<f32>> {
        let device = self.context.device();
        let queue = self.context.queue();

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (self.width * self.height * 16) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

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

        queue.submit(Some(encoder.finish()));

        let (tx, rx) = flume::bounded(1);
        let slice = staging_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::Maintain::Wait).panic_on_timeout();
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
