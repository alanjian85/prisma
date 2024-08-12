use std::error::Error;

use encase::{ShaderType, UniformBuffer};

use super::{Camera, RenderContext};

#[derive(Default, ShaderType)]
pub struct Scene {
    camera: Camera,
    env_map: u32,
}

impl Scene {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    pub fn set_env_map(&mut self, env_map: u32) {
        self.env_map = env_map;
    }

    fn as_wgsl_bytes(&self) -> encase::internal::Result<Vec<u8>> {
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer.write(self)?;
        Ok(buffer.into_inner())
    }

    pub fn build(
        &self,
        context: &RenderContext,
    ) -> Result<(wgpu::BindGroupLayout, wgpu::BindGroup), Box<dyn Error>> {
        let device = context.device();
        let queue = context.queue();

        let wgsl_bytes = &self.as_wgsl_bytes()?;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        queue.write_buffer(&uniform_buffer, 0, wgsl_bytes);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        Ok((bind_group_layout, bind_group))
    }
}
