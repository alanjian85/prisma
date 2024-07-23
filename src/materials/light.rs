use crate::core::{Material, RayIntersection, Texture2};
use palette::LinSrgb;
use std::sync::Arc;

pub struct Light {
    texture: Arc<dyn Texture2>,
}

impl Light {
    pub fn new(texture: Arc<dyn Texture2>) -> Self {
        Self { texture }
    }
}

impl Material for Light {
    fn emit(&self, intersection: &RayIntersection) -> LinSrgb<f64> {
        self.texture.sample(intersection.uv)
    }
}
