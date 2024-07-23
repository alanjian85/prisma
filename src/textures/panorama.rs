use crate::core::{Texture2, Texture3};
use crate::textures::ImageHdr;
use nalgebra::{Vector2, Vector3};
use palette::LinSrgb;
use std::{error, f64};

pub struct Panorama {
    image: ImageHdr,
}

impl Panorama {
    pub fn new(path: &str) -> Result<Self, Box<dyn error::Error + Send + Sync + 'static>> {
        Ok(Self {
            image: ImageHdr::new(path)?,
        })
    }
}

impl Texture3 for Panorama {
    fn sample(&self, uvw: Vector3<f64>) -> LinSrgb<f64> {
        let theta = (-uvw.y).acos();
        let phi = (-uvw.z).atan2(uvw.x) + f64::consts::PI;

        let u = phi / (2.0 * f64::consts::PI);
        let v = theta / f64::consts::PI;

        self.image.sample(Vector2::new(u, v))
    }
}
