use std::{error::Error, mem};

use encase::{ShaderType, UniformBuffer};

use super::RenderContext;

#[derive(Default, ShaderType)]
pub struct Scene {
    env_map: u32,
}

impl Scene {
    pub fn new() -> Self {
        Self { env_map: 0 }
    }

    pub fn set_env(&mut self, env_map: u32) {
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

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: mem::size_of::<Scene>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        queue.write_buffer(&uniform_buffer, 0, &self.as_wgsl_bytes()?);

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
