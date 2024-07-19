use crate::core::{Primitive, Ray, RayIntersection};
use std::ops::Range;

#[derive(Default)]
pub struct Scene {
    primitives: Vec<Box<dyn Primitive>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
        }
    }

    pub fn add(&mut self, primitive: Box<dyn Primitive>) {
        self.primitives.push(primitive);
    }

    pub fn intersect(&self, ray: &Ray, range: &Range<f64>) -> Option<RayIntersection> {
        let mut range = range.clone();
        let mut closest_intersection = None;
        for primitive in &self.primitives {
            if let Some((t, intersection)) = primitive.intersect(ray, &range) {
                range.end = t;
                closest_intersection = Some(intersection);
            }
        }
        closest_intersection
    }
}
