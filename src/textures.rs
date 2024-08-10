use std::rc::Rc;

use crate::core::RenderContext;

mod image_hdr;

pub use image_hdr::ImageHdr;

pub struct Textures<'a> {
    context: &'a RenderContext,
    env_map: Option<Rc<dyn Texture2>>,
}

pub trait Texture2 {
    fn texture(&self) -> &wgpu::Texture;
    fn view(&self) -> &wgpu::TextureView;
}

impl<'a> Textures<'a> {
    pub fn new(context: &'a RenderContext) -> Self {
        Self {
            context,
            env_map: None,
        }
    }

    pub fn set_env_map(&mut self, env_map: Rc<dyn Texture2>) {
        self.env_map = Some(env_map);
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
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(self.env_map.as_ref().unwrap().view()),
            }],
        });

        (bind_group_layout, bind_group)
    }
}
