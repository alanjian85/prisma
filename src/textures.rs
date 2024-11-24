use std::{error::Error, num::NonZeroU32, rc::Rc};

use gltf::image::Data;
use image::ImageReader;

use crate::render::RenderContext;

mod texture;
mod texture_hdr;

use self::{texture::Texture, texture_hdr::TextureHdr};

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

    pub fn load_texture_hdr(&mut self, path: &str) -> Result<u32, Box<dyn Error>> {
        let image = ImageReader::open(path)?.decode()?.into_rgba32f();
        let width = image.width();
        let height = image.height();
        self.registry.push(Rc::new(TextureHdr::new(
            &self.context,
            image.as_raw(),
            width,
            height,
        )?));
        Ok(self.registry.len() as u32 - 1)
    }

    pub fn add_texture(&mut self, image: &Data) -> u32 {
        let num_pixels = (image.width * image.height) as usize;
        let mut data = Vec::with_capacity(num_pixels * 4);

        match image.format {
            gltf::image::Format::R8G8B8 => {
                for i in 0..num_pixels {
                    data.push(image.pixels[3 * i]);
                    data.push(image.pixels[3 * i + 1]);
                    data.push(image.pixels[3 * i + 2]);
                    data.push(0xFF);
                }
            }
            gltf::image::Format::R8G8B8A8 => {
                for i in 0..num_pixels {
                    data.push(image.pixels[4 * i]);
                    data.push(image.pixels[4 * i + 1]);
                    data.push(image.pixels[4 * i + 2]);
                    data.push(image.pixels[4 * i + 3]);
                }
            }
            _ => todo!(),
        }

        self.registry.push(Rc::new(Texture::new(
            &self.context,
            &data,
            image.width,
            image.height,
        )));
        self.registry.len() as u32 - 1
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
