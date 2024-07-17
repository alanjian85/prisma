use crate::core::Ray;
use nalgebra::Vector3;

pub struct RayIntersection {
    pub normal: Vector3<f64>,
}

pub trait Intersect {
    fn intersect(&self, _ray: &Ray) -> Option<RayIntersection> {
        None
    }
}
