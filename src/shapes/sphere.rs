use crate::core::{Intersect, Ray, RayIntersection};
use nalgebra::Point3;
use std::ops::Range;

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
    fn intersect(&self, ray: &Ray, range: &Range<f64>) -> Option<RayIntersection> {
        let a = ray.dir.magnitude_squared();
        let b = ray.dir.dot(&(self.center - ray.orig));
        let c = (self.center - ray.orig).magnitude_squared() - self.radius * self.radius;
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return None;
        }

        let discriminant = discriminant.sqrt();
        let mut t = (b - discriminant) / a;
        if !range.contains(&t) {
            t = (b + discriminant) / a;
            if !range.contains(&t) {
                return None;
            }
        }

        let pos = ray.at(t);
        let normal = (pos - self.center) / self.radius;
        Some(RayIntersection { t, pos, normal })
    }
}
