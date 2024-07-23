use crate::core::{Material, Ray, RayIntersection, Texture2};
use crate::{math, utils};
use palette::LinSrgb;
use rand::prelude::*;
use std::sync::Arc;

pub struct Metal {
    texture: Arc<dyn Texture2>,
    fuzziness: f64,
}

impl Metal {
    pub fn new(texture: Arc<dyn Texture2>, fuzziness: f64) -> Self {
        Self { texture, fuzziness }
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        rng: &mut ThreadRng,
        ray: &Ray,
        intersection: &RayIntersection,
    ) -> Option<(Ray, LinSrgb<f64>)> {
        let dir = math::reflect(ray.dir.normalize(), intersection.normal);
        let dir = dir + self.fuzziness * utils::rand_unit_vec3(rng);
        let ray = Ray::new(intersection.pos, dir);
        if ray.dir.dot(&intersection.normal) < 0.0 {
            return None;
        }
        Some((ray, self.texture.sample(intersection.uv)))
    }
}
