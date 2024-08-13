use std::{error::Error, num::NonZeroU32, rc::Rc};

use crate::core::RenderContext;

mod image_hdr;

use image_hdr::ImageHdr;

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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(self.registry.len() as u32),
            }],
        });

        let view_array: Vec<_> = self.registry.iter().map(|texture| texture.view()).collect();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&view_array),
            }],
        });

        (bind_group_layout, bind_group)
    }
}
