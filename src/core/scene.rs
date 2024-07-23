use crate::core::{Primitive, Ray, RayIntersection, Texture3};
use nalgebra::Vector3;
use palette::LinSrgb;
use std::ops::Range;
use std::sync::Arc;

#[derive(Default)]
pub struct Scene {
    primitives: Vec<Arc<dyn Primitive>>,
    env_map: Option<Arc<dyn Texture3>>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            primitives: Vec::new(),
            env_map: None,
        }
    }

    pub fn add(&mut self, primitive: Arc<dyn Primitive>) {
        self.primitives.push(primitive);
    }

    pub fn set_env(&mut self, env_map: Arc<dyn Texture3>) {
        self.env_map = Some(env_map.clone());
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

    pub fn sample_env(&self, uvw: Vector3<f64>) -> LinSrgb<f64> {
        if let Some(env_map) = &self.env_map {
            return env_map.sample(uvw);
        }
        LinSrgb::new(0.0, 0.0, 0.0)
    }
}
