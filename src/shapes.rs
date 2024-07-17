use crate::{Intersect, Ray};
use nalgebra::Point3;

pub struct Sphere {
    center: Point3<f64>,
    radius: f64,
}

impl Sphere {
    pub fn new(center: Point3<f64>, radius: f64) -> Self {
        Self { center, radius }
    }
}

impl Intersect for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let a = ray.dir.magnitude_squared();
        let b = ray.dir.dot(&(self.center - ray.orig));
        let c = (self.center - ray.orig).magnitude_squared() - self.radius * self.radius;
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        Some((b - discriminant.sqrt()) / a)
    }
}
