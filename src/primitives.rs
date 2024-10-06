use encase::StorageBuffer;
use glam::{Vec2, Vec3};
use gltf::{buffer::Data, Primitive};

use crate::{
    core::{Triangle, Vertex},
    render::RenderContext,
};

#[derive(Default)]
pub struct Primitives {
    vertices: Vec<Vertex>,
    offsets: Vec<u32>,
    material_indices: Vec<u32>,
}

impl Primitives {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertex(&self, primitive: u32, vertex: u32) -> Vertex {
        self.vertices[(vertex + self.offsets[primitive as usize]) as usize]
    }

    pub fn add(
        &mut self,
        buffers: &[Data],
        primitive: &Primitive,
        material_idx: u32,
    ) -> Option<Vec<Triangle>> {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        let positions: Vec<_> = reader.read_positions()?.collect();
        let normals: Vec<_> = reader.read_normals()?.collect();
        let tex_coords: Vec<_> = reader.read_tex_coords(0)?.into_f32().collect();
        let mut vertices = Vec::with_capacity(positions.len());
        for i in 0..positions.len() {
            vertices.push(Vertex {
                pos: Vec3::from_array(positions[i]),
                normal: Vec3::from_array(normals[i]),
                tex_coord: Vec2::from_array(tex_coords[i]),
            });
        }

        let primitive = self.offsets.len() as u32;
        let offset = self.vertices.len() as u32;
        self.vertices.append(&mut vertices);
        self.offsets.push(offset);
        self.material_indices.push(material_idx);

        let indices: Vec<_> = reader.read_indices()?.into_u32().collect();
        let mut triangles = Vec::new();
        for i in 0..indices.len() / 3 {
            triangles.push(Triangle {
                primitive,
                v0: indices[3 * i],
                v1: indices[3 * i + 1],
                v2: indices[3 * i + 2],
            });
        }

        Some(triangles)
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

        let material_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&material_buffer, 0, &wgsl_bytes);

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
                    resource: material_buffer.as_entire_binding(),
                },
            ],
        });

        Ok((bind_group_layout, bind_group))
    }
}
