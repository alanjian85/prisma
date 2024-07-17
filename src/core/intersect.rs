use crate::core::Ray;
use nalgebra::Vector3;
use std::ops::Range;

pub struct RayIntersection {
    pub t: f64,
    pub normal: Vector3<f64>,
}

pub trait Intersect {
    fn intersect(&self, _ray: &Ray, _range: &Range<f64>) -> Option<RayIntersection> {
        None
    }
}
