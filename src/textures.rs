use std::{error::Error, rc::Rc};

use crate::core::RenderContext;

mod image_hdr;

pub use image_hdr::ImageHdr;

pub struct Textures {
    context: Rc<RenderContext>,
    registry: Vec<Rc<dyn Texture2>>,
}

pub trait Texture2 {
    fn texture(&self) -> &wgpu::Texture;
    fn view(&self) -> &wgpu::TextureView;
}

impl Textures {
    pub fn new(context: Rc<RenderContext>) -> Self {
        Self {
            context,
            registry: Vec::new(),
        }
    }

    pub fn create_image_hdr(&mut self, path: &str) -> Result<u32, Box<dyn Error + Send + Sync>> {
        self.registry
            .push(Rc::new(ImageHdr::new(&self.context, path)?));
        Ok(self.registry.len() as u32 - 1)
    }

    pub fn build(&self) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
        let device = self.context.device();

        let entries: Vec<_> = self
            .registry
            .iter()
            .enumerate()
            .map(|(i, _)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            })
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });

        let entries: Vec<_> = self
            .registry
            .iter()
            .enumerate()
            .map(|(i, texture)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::TextureView(texture.view()),
            })
            .collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &entries,
        });

        (bind_group_layout, bind_group)
    }
}
