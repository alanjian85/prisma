use std::{cell::RefCell, rc::Rc};

use encase::{ShaderType, StorageBuffer, UniformBuffer};

use crate::{core::Primitive, models::Models, render::RenderContext};

use self::bvh::Bvh;

mod bvh;
mod camera;

pub use camera::{Camera, CameraBuilder};

#[derive(Default, ShaderType)]
struct Uniform {
    camera: Camera,
    env_map: u32,
}

pub struct Scene {
    models: Rc<RefCell<Models>>,
    uniform: Uniform,
    primitives: Vec<Primitive>,
}

impl Scene {
    pub fn new(models: Rc<RefCell<Models>>) -> Self {
        Self {
            models,
            uniform: Uniform::default(),
            primitives: Vec::new(),
        }
    }

    pub fn set_camera(&mut self, camera: Camera) -> &mut Self {
        self.uniform.camera = camera;
        self
    }

    pub fn set_env_map(&mut self, env_map: u32) -> &mut Self {
        self.uniform.env_map = env_map;
        self
    }

    pub fn add(&mut self, idx: u32) -> &mut Self {
        self.primitives
            .append(&mut self.models.borrow().primitives(idx).clone());
        self
    }

    pub fn build(
        &mut self,
        context: &RenderContext,
    ) -> encase::internal::Result<(wgpu::BindGroupLayout, wgpu::BindGroup)> {
        let device = context.device();
        let queue = context.queue();

        let mut wgsl_bytes = UniformBuffer::new(Vec::new());
        wgsl_bytes.write(&self.uniform)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buffer, 0, &wgsl_bytes);

        let bvh = Bvh::new(self.models.borrow().meshes(), &mut self.primitives);

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&self.primitives)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let primitive_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&primitive_buffer, 0, &wgsl_bytes);

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&bvh.flatten())?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let bvh_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&bvh_buffer, 0, &wgsl_bytes);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: primitive_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bvh_buffer.as_entire_binding(),
                },
            ],
        });

        Ok((bind_group_layout, bind_group))
    }
}
