use crate::core::{Material, Ray, RayIntersection};
use crate::math;
use palette::LinSrgb;
use rand::rngs::ThreadRng;

pub struct Metal {
    albedo: LinSrgb<f64>,
}

impl Metal {
    pub fn new(albedo: LinSrgb<f64>) -> Self {
        Self { albedo }
    }
}

impl Material for Metal {
    fn scatter(
        &self,
        _rng: &mut ThreadRng,
        ray: &Ray,
        intersection: &RayIntersection,
    ) -> (Ray, LinSrgb<f64>) {
        let dir = math::reflect(ray.dir, intersection.normal);
        let ray = Ray::new(intersection.pos, dir);
        (ray, self.albedo)
    }
}
