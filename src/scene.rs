use std::rc::Rc;

use encase::{ShaderType, StorageBuffer, UniformBuffer};
use gltf::{buffer, image, Node};

use crate::{
    core::Triangle, materials::Materials, primitives::Primitives, render::RenderContext,
    textures::Textures,
};

use self::bvh::Bvh;

mod bvh;
mod camera;

pub use camera::{Camera, CameraBuilder};

#[derive(Default, ShaderType)]
struct Uniform {
    camera: Camera,
    hdri: u32,
}

pub struct Scene {
    pub primitives: Primitives,
    pub materials: Materials,
    pub textures: Textures,
    uniform: Uniform,
    triangles: Vec<Triangle>,
}

impl Scene {
    pub fn new(context: Rc<RenderContext>) -> Self {
        Self {
            primitives: Primitives::new(),
            materials: Materials::new(),
            textures: Textures::new(context),
            uniform: Uniform::default(),
            triangles: Vec::new(),
        }
    }

    pub fn set_camera(&mut self, camera: Camera) -> &mut Self {
        self.uniform.camera = camera;
        self
    }

    pub fn set_hdri(&mut self, hdri: u32) -> &mut Self {
        self.uniform.hdri = hdri;
        self
    }

    pub fn load(&mut self, scene: &gltf::Scene, buffers: &[buffer::Data], images: &[image::Data]) {
        for image in images {
            self.textures.add_texture(image);
        }

        for node in scene.nodes() {
            self.load_node(node, buffers);
        }
    }

    fn load_node(&mut self, node: Node, buffers: &[buffer::Data]) {
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                self.triangles.append(
                    &mut self
                        .primitives
                        .add(
                            buffers,
                            &primitive,
                            self.materials.add(&primitive.material()).unwrap(),
                        )
                        .unwrap(),
                );
            }
        }

        for child in node.children() {
            self.load_node(child, buffers);
        }
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

        let bvh = Bvh::new(&self.primitives, &mut self.triangles);

        let mut wgsl_bytes = StorageBuffer::new(Vec::new());
        wgsl_bytes.write(&self.triangles)?;
        let wgsl_bytes = wgsl_bytes.into_inner();

        let triangle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: wgsl_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        queue.write_buffer(&triangle_buffer, 0, &wgsl_bytes);

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
                    resource: triangle_buffer.as_entire_binding(),
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
