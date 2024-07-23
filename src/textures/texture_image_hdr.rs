use image::{ImageReader, Rgb32FImage};
use nalgebra::Vector2;
use palette::LinSrgb;
use std::error;

pub struct TextureImageHdr {
    image: Rgb32FImage,
}

impl TextureImageHdr {
    pub fn new(path: &str) -> Result<Self, Box<dyn error::Error>> {
        let image = ImageReader::open(path)?.decode()?.into_rgb32f();
        Ok(Self { image })
    }

    pub fn sample(&self, uv: Vector2<f64>) -> LinSrgb<f64> {
        let u = uv.x.clamp(0.0, 0.999);
        let v = (1.0 - uv.y).clamp(0.0, 0.999);

        let x = (u * self.image.width() as f64) as u32;
        let y = (v * self.image.height() as f64) as u32;
        let rgb = self.image.get_pixel(x, y);

        let r = rgb.0[0] as f64;
        let g = rgb.0[1] as f64;
        let b = rgb.0[2] as f64;

        LinSrgb::new(r, g, b)
    }
}
