use crate::core::{Ray, RayIntersection};
use palette::LinSrgb;
use rand::prelude::*;

pub trait Material: Send + Sync {
    fn scatter(
        &self,
        _rng: &mut ThreadRng,
        _ray: &Ray,
        _intersection: &RayIntersection,
    ) -> Option<(Ray, LinSrgb<f64>)> {
        None
    }

    fn emit(&self, _intersection: &RayIntersection) -> LinSrgb<f64> {
        LinSrgb::new(0.0, 0.0, 0.0)
    }
}
