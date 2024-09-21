use encase::StorageBuffer;
use glam::Vec3;
use tobj::Mesh;

use crate::{
    core::{Primitive, Vertex},
    render::RenderContext,
};

#[derive(Default)]
pub struct Meshes {
    vertices: Vec<Vertex>,
    offsets: Vec<u32>,
    material_indices: Vec<u32>,
}

impl Meshes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertex(&self, idx: u32, vertex: u32) -> Vertex {
        self.vertices[(vertex + self.offsets[idx as usize]) as usize]
    }

    pub fn add(&mut self, mesh: &Mesh, material_idx_start: u32) -> Vec<Primitive> {
        let mut vertices = Vec::with_capacity(mesh.positions.len() / 3);
        for v in 0..mesh.positions.len() / 3 {
            vertices.push(Vertex {
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

        self.offsets.push(self.vertices.len() as u32);
        self.material_indices.push(material_idx_start); // + mesh.material_id.unwrap() as u32);
        self.vertices.append(&mut vertices);
        let idx = self.offsets.len() as u32 - 1;

        let mut primitives = Vec::new();
        for i in 0..mesh.indices.len() / 3 {
            primitives.push(Primitive {
                idx,
                v0: mesh.indices[3 * i],
                v1: mesh.indices[3 * i + 1],
                v2: mesh.indices[3 * i + 2],
            });
        }
        primitives
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

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&self.material_indices)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let material_idx_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&material_idx_buffer, 0, &wgsl_bytes);

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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: material_idx_buffer.as_entire_binding(),
                },
            ],
        });

        Ok((bind_group_layout, bind_group))
    }
}
