use encase::StorageBuffer;
use glam::Vec3;
use tobj::{LoadError, LoadOptions};

use crate::{
    core::{Primitive, Vertex},
    render::RenderContext,
};

#[derive(Default)]
pub struct Meshes {
    pub vertices: Vec<Vertex>,
    pub offsets: Vec<u32>,
}

impl Meshes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_model(&mut self, path: &str) -> Result<Vec<Primitive>, LoadError> {
        let (models, _materials) = tobj::load_obj(
            path,
            &LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )?;

        let mut primitives = Vec::new();
        let mut offset = self.vertices.len() as u32;

        for model in models {
            let mesh = model.mesh;
            for v in 0..mesh.positions.len() / 3 {
                self.vertices.push(Vertex {
                    pos: Vec3::new(
                        mesh.positions[3 * v],
                        mesh.positions[3 * v + 1],
                        mesh.positions[3 * v + 2],
                    ),
                    normal: Vec3::new(
                        mesh.normals[3 * v],
                        mesh.normals[3 * v + 1],
                        mesh.normals[3 * v + 2],
                    ),
                });
            }

            let idx = self.offsets.len() as u32;
            for i in 0..mesh.indices.len() / 3 {
                primitives.push(Primitive {
                    idx,
                    p0: mesh.indices[3 * i],
                    p1: mesh.indices[3 * i + 1],
                    p2: mesh.indices[3 * i + 2],
                });
            }

            self.offsets.push(offset);
            offset = self.vertices.len() as u32;
        }

        Ok(primitives)
    }

    pub fn build(
        &self,
        context: &RenderContext,
    ) -> encase::internal::Result<(wgpu::BindGroupLayout, wgpu::BindGroup)> {
        let device = context.device();
        let queue = context.queue();

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&self.vertices)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&vertex_buffer, 0, &wgsl_bytes);

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&self.offsets)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let offset_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&offset_buffer, 0, &wgsl_bytes);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: offset_buffer.as_entire_binding(),
                },
            ],
        });

        Ok((bind_group_layout, bind_group))
    }
}
