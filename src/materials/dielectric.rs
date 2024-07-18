use crate::core::{Material, Ray, RayIntersection};
use crate::math;
use palette::LinSrgb;
use rand::{rngs::ThreadRng, Rng};

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
        rng: &mut ThreadRng,
        ray: &Ray,
        intersection: &RayIntersection,
    ) -> Option<(Ray, LinSrgb<f64>)> {
        let eta = if intersection.front {
            self.eta.recip()
        } else {
            self.eta
        };

        let dir = ray.dir.normalize();
        let cosine = -dir.dot(&intersection.normal);
        let sine = (1.0 - cosine * cosine).sqrt();

        let rand = rng.gen_range(0.0..1.0);
        let dir = if eta * sine > 1.0 || rand < math::reflectance(cosine, eta) {
            math::reflect(dir, intersection.normal)
        } else {
            math::refract(dir, intersection.normal, eta)
        };
        let ray = Ray::new(intersection.pos, dir);
        Some((ray, LinSrgb::new(1.0, 1.0, 1.0)))
    }
}
