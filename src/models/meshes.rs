use encase::StorageBuffer;

use crate::{core::Vertex, render::RenderContext};

#[derive(Default)]
pub struct Meshes {
    vertices: Vec<Vertex>,
    offsets: Vec<u32>,
}

impl Meshes {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_vertex(&self, idx: u32, vertex: u32) -> Vertex {
        self.vertices[(vertex + self.offsets[idx as usize]) as usize]
    }

    pub fn add_mesh(&mut self, vertices: &mut Vec<Vertex>) -> u32 {
        self.offsets.push(self.vertices.len() as u32);
        self.vertices.append(vertices);
        self.offsets.len() as u32 - 1
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
