use nalgebra::{Vector2, Vector3};
use palette::LinSrgb;

pub trait Texture2: Send + Sync {
    fn sample(&self, uv: Vector2<f64>) -> LinSrgb<f64>;
}

pub trait Texture3: Send + Sync {
    fn sample(&self, uvw: Vector3<f64>) -> LinSrgb<f64>;
}
