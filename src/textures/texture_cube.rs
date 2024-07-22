use crate::textures::TextureImage;
use nalgebra::{Vector2, Vector3};
use palette::LinSrgb;
use std::error;

pub struct TextureCube {
    right: TextureImage,
    left: TextureImage,
    top: TextureImage,
    bottom: TextureImage,
    back: TextureImage,
    front: TextureImage,
}

impl TextureCube {
    pub fn new(paths: TextureCubePaths) -> Result<Self, Box<dyn error::Error>> {
        Ok(Self {
            right: TextureImage::new(paths.right)?,
            left: TextureImage::new(paths.left)?,
            top: TextureImage::new(paths.top)?,
            bottom: TextureImage::new(paths.bottom)?,
            back: TextureImage::new(paths.back)?,
            front: TextureImage::new(paths.front)?,
        })
    }

    pub fn sample(&self, uvw: Vector3<f64>) -> LinSrgb<f64> {
        let abs = uvw.abs();
        let (u, v, m);
        let texture = if abs.x > abs.y && abs.x > abs.z {
            m = abs.x;
            if uvw.x > 0.0 {
                u = uvw.z;
                v = uvw.y;
                &self.right
            } else {
                u = -uvw.z;
                v = uvw.y;
                &self.left
            }
        } else if abs.y > abs.z {
            m = abs.y;
            if uvw.y > 0.0 {
                u = uvw.x;
                v = uvw.z;
                &self.top
            } else {
                u = uvw.x;
                v = -uvw.z;
                &self.bottom
            }
        } else {
            m = abs.z;
            if uvw.z > 0.0 {
                u = -uvw.x;
                v = uvw.y;
                &self.back
            } else {
                u = uvw.x;
                v = uvw.y;
                &self.front
            }
        };
        let u = 0.5 * (u / m + 1.0);
        let v = 0.5 * (v / m + 1.0);
        texture.sample(Vector2::new(u, v))
    }
}

pub struct TextureCubePaths<'a> {
    pub right: &'a str,
    pub left: &'a str,
    pub top: &'a str,
    pub bottom: &'a str,
    pub back: &'a str,
    pub front: &'a str,
}
