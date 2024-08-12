use std::{error::Error, slice};

use image::ImageReader;

use crate::core::RenderContext;

use super::Texture2;

pub struct ImageHdr {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl ImageHdr {
    pub fn new(context: &RenderContext, path: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let device = context.device();
        let queue = context.queue();

        let image = ImageReader::open(path)?.decode()?.into_rgba32f();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            unsafe { slice::from_raw_parts(image.as_ptr() as *const u8, image.len() * 4) },
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(image.width() * 16),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: image.width(),
                height: image.height(),
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Ok(Self { texture, view })
    }
}

impl Texture2 for ImageHdr {
    fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
}
