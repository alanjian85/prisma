use std::{collections::HashMap, rc::Rc};

use indicatif::ProgressBar;

use crate::config::{Config, Size};

use super::RenderContext;

pub struct Renderer {
    context: Rc<RenderContext>,
    width: u32,
    height: u32,
    samples: u32,
    target_bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    render_target: wgpu::Texture,
}

pub struct BindGroupLayoutSet {
    pub textures: wgpu::BindGroupLayout,
    pub materials: wgpu::BindGroupLayout,
    pub scene: wgpu::BindGroupLayout,
}

pub struct BindGroupSet {
    pub textures: wgpu::BindGroup,
    pub materials: wgpu::BindGroup,
    pub scene: wgpu::BindGroup,
}

impl Renderer {
    pub fn new(
        context: Rc<RenderContext>,
        config: &Config,
        bind_group_layout_set: BindGroupLayoutSet,
    ) -> Self {
        let device = context.device();

        let Size { width, height } = config.size;
        let width = (width + 15) / 16 * 16;
        let height = (height + 15) / 16 * 16;

        let target_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                &target_bind_group_layout,
                &bind_group_layout_set.textures,
                &bind_group_layout_set.materials,
                &bind_group_layout_set.scene,
            ],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..4,
            }],
        });

        let shader_module =
            device.create_shader_module(wgpu::include_wgsl!("../../shaders/render.wgsl"));

        let mut constants = HashMap::new();
        constants.insert(String::from("MAX_DEPTH"), config.depth as f64);

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
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        Self {
            context,
            width,
            height,
            samples: config.samples,
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

        let progress_bar = ProgressBar::new(self.samples as u64);

        for sample in 0..self.samples {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

            {
                let sample: [u8; 4] = sample.to_ne_bytes();

                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&self.pipeline);
                compute_pass.set_bind_group(0, &output_bind_group, &[]);
                compute_pass.set_bind_group(1, &bind_group_set.textures, &[]);
                compute_pass.set_bind_group(2, &bind_group_set.materials, &[]);
                compute_pass.set_bind_group(3, &bind_group_set.scene, &[]);
                compute_pass.set_push_constants(0, &sample);
                compute_pass.dispatch_workgroups(self.width / 16, self.height / 16, 1);
            }

            queue.submit(Some(encoder.finish()));
            progress_bar.inc(1);
        }

        device.poll(wgpu::Maintain::Wait);
        progress_bar.finish_and_clear();
    }

    pub fn render_target(&self) -> &wgpu::Texture {
        &self.render_target
    }
}
