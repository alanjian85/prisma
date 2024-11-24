use encase::{ShaderType, StorageBuffer};

use crate::render::RenderContext;

#[derive(ShaderType)]
pub struct Material {
    base_color_texture: u32,
    metallic_roughness_texture: u32,
    normal_texture: u32,
    emissive_texture: u32,
}

#[derive(Default)]
pub struct Materials {
    registry: Vec<Material>,
}

impl Materials {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, material: &gltf::Material) -> Option<u32> {
        let pbr_metallic_roughness = material.pbr_metallic_roughness();
        let base_color_texture = pbr_metallic_roughness
            .base_color_texture()?
            .texture()
            .source()
            .index() as u32;
        let metallic_roughness_texture = pbr_metallic_roughness
            .metallic_roughness_texture()?
            .texture()
            .source()
            .index() as u32;
        let normal_texture = material.normal_texture()?.texture().source().index() as u32;
        //        let emissive_texture = material.emissive_texture()?.texture().source().index() as u32;

        self.registry.push(Material {
            base_color_texture,
            metallic_roughness_texture,
            normal_texture,
            emissive_texture: 0, // emissive_texture,
        });
        Some(self.registry.len() as u32 - 1)
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
