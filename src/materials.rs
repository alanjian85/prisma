use std::rc::Rc;

use encase::{ShaderType, StorageBuffer};
use glam::Vec3;

use crate::core::RenderContext;

#[derive(Default, ShaderType)]
pub struct Material {
    ty: u32,
    albedo: Vec3,
    fuzziness: f32,
    ior: f32,
}

pub struct Materials {
    context: Rc<RenderContext>,
    registry: Vec<Material>,
}

impl Materials {
    pub fn new(context: Rc<RenderContext>) -> Self {
        Self {
            context,
            registry: Vec::new(),
        }
    }

    pub fn create_lambertian(&mut self, albedo: Vec3) -> u32 {
        self.registry.push(Material {
            ty: 0,
            albedo,
            ..Default::default()
        });
        self.registry.len() as u32 - 1
    }

    pub fn create_metal(&mut self, albedo: Vec3, fuzziness: f32) -> u32 {
        self.registry.push(Material {
            ty: 1,
            albedo,
            fuzziness,
            ..Default::default()
        });
        self.registry.len() as u32 - 1
    }

    pub fn create_dielectric(&mut self, ior: f32) -> u32 {
        self.registry.push(Material {
            ty: 2,
            ior,
            ..Default::default()
        });
        self.registry.len() as u32 - 1
    }

    pub fn build(&self) -> encase::internal::Result<(wgpu::BindGroupLayout, wgpu::BindGroup)> {
        let device = self.context.device();
        let queue = self.context.queue();

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&self.registry)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let storage_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&storage_buffer, 0, &wgsl_bytes);

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
                resource: storage_buffer.as_entire_binding(),
            }],
        });

        Ok((bind_group_layout, bind_group))
    }
}
