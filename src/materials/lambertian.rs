use crate::core::{Material, Ray, RayIntersection};
use crate::utils;
use palette::LinSrgb;
use rand::rngs::ThreadRng;

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
    ) -> (Ray, LinSrgb<f64>) {
        let dir = intersection.normal + utils::rand_unit_vec3(rng);
        let ray = Ray::new(intersection.pos, dir);
        (ray, self.albedo)
    }
}
