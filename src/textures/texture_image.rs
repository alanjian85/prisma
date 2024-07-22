use image::{ImageReader, RgbImage};
use nalgebra::Vector2;
use palette::{LinSrgb, Srgb};
use std::error;

pub struct TextureImage {
    image: RgbImage,
}

impl TextureImage {
    pub fn new(path: &str) -> Result<Self, Box<dyn error::Error>> {
        let image = ImageReader::open(path)?.decode()?.into_rgb8();
        Ok(Self { image })
    }

    pub fn sample(&self, uv: Vector2<f64>) -> LinSrgb<f64> {
        let u = uv.x.clamp(0.0, 0.999);
        let v = (1.0 - uv.y).clamp(0.0, 0.999);

        let x = (u * self.image.width() as f64) as u32;
        let y = (v * self.image.height() as f64) as u32;
        let rgb = self.image.get_pixel(x, y);

        let r = rgb.0[0] as f64 / 255.0;
        let g = rgb.0[1] as f64 / 255.0;
        let b = rgb.0[2] as f64 / 255.0;

        Srgb::new(r, g, b).into_linear()
    }
}
