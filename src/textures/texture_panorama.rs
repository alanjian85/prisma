use crate::textures::TextureImageHdr;
use nalgebra::{Vector2, Vector3};
use palette::LinSrgb;
use std::{error, f64};

pub struct TexturePanorama {
    image: TextureImageHdr,
}

impl TexturePanorama {
    pub fn new(path: &str) -> Result<Self, Box<dyn error::Error>> {
        Ok(Self {
            image: TextureImageHdr::new(path)?,
        })
    }

    pub fn sample(&self, p: Vector3<f64>) -> LinSrgb<f64> {
        let theta = (-p.y).acos();
        let phi = (-p.z).atan2(p.x) + f64::consts::PI;

        let u = phi / (2.0 * f64::consts::PI);
        let v = theta / f64::consts::PI;

        self.image.sample(Vector2::new(u, v))
    }
}
