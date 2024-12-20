use std::rc::Rc;

use encase::{ShaderType, StorageBuffer, UniformBuffer};
use glam::{Mat4, Quat, Vec3};
use gltf::{buffer, camera::Projection, image, scene, Node};

use crate::{
    config::Config, core::Triangle, materials::Materials, primitives::Primitives,
    render::RenderContext, textures::Textures,
};

use self::bvh::Bvh;

mod bvh;
mod camera;

pub use camera::{Camera, CameraBuilder};

pub struct Scene {
    pub primitives: Primitives,
    pub materials: Materials,
    pub textures: Textures,
    uniform: Uniform,
    triangles: Vec<Triangle>,
}

#[derive(Default, ShaderType)]
struct Uniform {
    camera: Camera,
    hdri: u32,
}

pub struct Transform {
    pub transform: Mat4,
    pub inv_trans: Mat4,
}

impl Transform {
    pub fn new(transform: Mat4) -> Self {
        Self {
            transform,
            inv_trans: transform.inverse().transpose(),
        }
    }
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

    pub fn set_hdri(&mut self, hdri: u32) -> &mut Self {
        self.uniform.hdri = hdri;
        self
    }

    pub fn load(
        &mut self,
        config: &Config,
        scene: &gltf::Scene,
        buffers: &[buffer::Data],
        images: &[image::Data],
    ) {
        for image in images {
            self.textures.add_texture(image);
        }

        for node in scene.nodes() {
            self.load_node(node, config, buffers, &Mat4::IDENTITY);
        }
    }

    fn load_node(
        &mut self,
        node: Node,
        config: &Config,
        buffers: &[buffer::Data],
        parent_transform: &Mat4,
    ) {
        let transform_matrix = *parent_transform * transform_to_matrix(&node.transform());
        let transform = Transform::new(transform_matrix);

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                let material_idx = self.materials.add(&primitive.material()).unwrap();
                self.triangles.append(
                    &mut self
                        .primitives
                        .add(buffers, &primitive, &transform, material_idx)
                        .unwrap(),
                );
            }
        }

        if let Some(camera) = node.camera() {
            match camera.projection() {
                Projection::Perspective(perspective) => {
                    let mut camera_builder = CameraBuilder::new();
                    camera_builder
                        .transform(transform_matrix)
                        .yfov(perspective.yfov());
                    self.uniform.camera =
                        camera_builder.build(config.size.width, config.size.height);
                }
                _ => todo!(),
            }
        }

        for child in node.children() {
            self.load_node(child, config, buffers, &transform_matrix);
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

fn transform_to_matrix(transform: &scene::Transform) -> Mat4 {
    match transform {
        scene::Transform::Matrix { matrix } => Mat4::from_cols_array_2d(matrix),
        scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => Mat4::from_scale_rotation_translation(
            Vec3::from_array(*scale),
            Quat::from_array(*rotation),
            Vec3::from_array(*translation),
        ),
    }
}
