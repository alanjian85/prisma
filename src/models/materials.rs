use encase::{ShaderType, StorageBuffer};
use glam::Vec3;

use crate::render::RenderContext;

#[derive(ShaderType)]
struct Material {
    diffuse: Vec3,
    emission: Vec3,
}

#[derive(Default)]
pub struct Materials {
    registry: Vec<Material>,
}

impl Materials {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, material: &tobj::Material) {
        let emission: [f32; 3] = material.unknown_param["Ke"]
            .split(' ')
            .map(|elem| elem.parse().unwrap())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        self.registry.push(Material {
            diffuse: Vec3::from_array(material.diffuse.unwrap()),
            emission: Vec3::from_array(emission),
        });
    }

    pub fn len(&self) -> usize {
        self.registry.len()
    }

    pub fn build(
        &self,
        context: &RenderContext,
    ) -> encase::internal::Result<(wgpu::BindGroupLayout, wgpu::BindGroup)> {
        let device = context.device();
        let queue = context.queue();

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&self.registry)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, &wgsl_bytes);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                resource: buffer.as_entire_binding(),
            }],
        });

        Ok((bind_group_layout, bind_group))
    }
}
