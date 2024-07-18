use crate::core::{Material, Ray};
use nalgebra::{Point3, Vector3};
use std::ops::Range;
use std::rc::Rc;

pub struct RayIntersection {
    pub pos: Point3<f64>,
    pub front: bool,
    pub normal: Vector3<f64>,
    pub material: Rc<dyn Material>,
}

pub fn flip_normal(dir: Vector3<f64>, normal: Vector3<f64>) -> (bool, Vector3<f64>) {
    if dir.dot(&normal) < 0.0 {
        (true, normal)
    } else {
        (false, -normal)
    }
}

pub trait Primitive {
    fn intersect(&self, _ray: &Ray, _range: &Range<f64>) -> Option<(f64, RayIntersection)> {
        None
    }
}
