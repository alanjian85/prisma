use crate::core::{Material, Primitive, Ray, RayIntersection};
use nalgebra::Point3;
use std::ops::Range;
use std::rc::Rc;

pub struct Sphere {
    center: Point3<f64>,
    radius: f64,
    material: Rc<dyn Material>,
}

impl Sphere {
    pub fn new(center: Point3<f64>, radius: f64, material: Rc<dyn Material>) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Primitive for Sphere {
    fn intersect(&self, ray: &Ray, range: &Range<f64>) -> Option<(f64, RayIntersection)> {
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
        Some((
            t,
            RayIntersection {
                pos,
                normal,
                material: self.material.clone(),
            },
        ))
    }
}