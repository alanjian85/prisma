use crate::core::{Material, Ray, RayIntersection};
use crate::math;
use palette::LinSrgb;
use rand::rngs::ThreadRng;

pub struct Dielectric {
    eta: f64,
}

impl Dielectric {
    pub fn new(eta: f64) -> Self {
        Self { eta }
    }
}

impl Material for Dielectric {
    fn scatter(
        &self,
        _rng: &mut ThreadRng,
        ray: &Ray,
        intersection: &RayIntersection,
    ) -> Option<(Ray, LinSrgb<f64>)> {
        let eta = if intersection.front {
            self.eta.recip()
        } else {
            self.eta
        };
        let dir = math::refract(ray.dir.normalize(), intersection.normal, eta);
        let ray = Ray::new(intersection.pos, dir);
        Some((ray, LinSrgb::new(1.0, 1.0, 1.0)))
    }
}
