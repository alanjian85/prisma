use crate::core::{Material, Ray, RayIntersection, Texture2};
use crate::utils;
use palette::LinSrgb;
use rand::prelude::*;
use std::sync::Arc;

pub struct Lambertian {
    texture: Arc<dyn Texture2>,
}

impl Lambertian {
    pub fn new(texture: Arc<dyn Texture2>) -> Self {
        Self { texture }
    }
}

impl Material for Lambertian {
    fn scatter(
        &self,
        rng: &mut ThreadRng,
        _ray: &Ray,
        intersection: &RayIntersection,
    ) -> Option<(Ray, LinSrgb<f64>)> {
        let mut dir = intersection.normal + utils::rand_unit_vec3(rng);
        if utils::is_vec3_near_zero(dir) {
            dir = intersection.normal;
        }
        let ray = Ray::new(intersection.pos, dir);
        Some((ray, self.texture.sample(intersection.uv)))
    }
}
