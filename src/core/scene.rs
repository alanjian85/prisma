use crate::core::{Intersect, Ray, RayIntersection};
use std::ops::Range;

pub struct Scene {
    objects: Vec<Box<dyn Intersect>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    pub fn add(&mut self, object: Box<dyn Intersect>) {
        self.objects.push(object);
    }
}

impl Intersect for Scene {
    fn intersect(&self, ray: &Ray, range: &Range<f64>) -> Option<RayIntersection> {
        let mut range = range.clone();
        let mut closest_intersection = None;
        for object in &self.objects {
            if let Some(intersection) = object.intersect(ray, &range) {
                range.end = intersection.t;
                closest_intersection = Some(intersection);
            }
        }
        closest_intersection
    }
}
