use crate::core::{Material, Primitive, Ray, RayIntersection};
use crate::utils;
use nalgebra::{Point3, Vector2, Vector3};
use std::ops::Range;
use std::sync::Arc;

pub struct Quad {
    orig: Point3<f64>,
    u: Vector3<f64>,
    v: Vector3<f64>,
    normal: Vector3<f64>,
    dist: f64,
    factor: Vector3<f64>,
    material: Arc<dyn Material>,
}

impl Quad {
    pub fn new(
        orig: Point3<f64>,
        u: Vector3<f64>,
        v: Vector3<f64>,
        material: Arc<dyn Material>,
    ) -> Self {
        let normal = u.cross(&v);
        let factor = normal / normal.dot(&normal);
        let normal = normal.normalize();
        let dist = orig.coords.dot(&normal);
        Self {
            orig,
            u,
            v,
            normal,
            dist,
            factor,
            material,
        }
    }
}

impl Primitive for Quad {
    fn intersect(&self, ray: &Ray, range: &Range<f64>) -> Option<(f64, RayIntersection)> {
        let denom = ray.dir.dot(&self.normal);
        if utils::is_near_zero(denom) {
            return None;
        }

        let t = (self.dist - ray.orig.coords.dot(&self.normal)) / denom;
        if !range.contains(&t) {
            return None;
        }
        let pos = ray.at(t);

        let p = pos - self.orig;
        let u = self.factor.dot(&p.cross(&self.v));
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let v = self.factor.dot(&self.u.cross(&p));
        if !(0.0..=1.0).contains(&v) {
            return None;
        }

        let normal = self.normal;
        let (front, normal) = RayIntersection::flip_normal(ray.dir, normal);
        let uv = Vector2::new(u, v);
        Some((
            t,
            RayIntersection {
                pos,
                front,
                normal,
                uv,
                material: self.material.clone(),
            },
        ))
    }
}
