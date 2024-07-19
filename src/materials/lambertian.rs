use crate::core::{Material, Ray, RayIntersection};
use crate::utils;
use palette::LinSrgb;
use rand::prelude::*;

pub struct Lambertian {
    albedo: LinSrgb<f64>,
}

impl Lambertian {
    pub fn new(albedo: LinSrgb<f64>) -> Self {
        Self { albedo }
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
        Some((ray, self.albedo))
    }
}
