use crate::core::{Ray, RayIntersection};
use palette::LinSrgb;
use rand::prelude::*;

pub trait Material: Send + Sync {
    fn scatter(
        &self,
        rng: &mut ThreadRng,
        ray: &Ray,
        intersection: &RayIntersection,
    ) -> Option<(Ray, LinSrgb<f64>)>;
}
