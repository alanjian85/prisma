use crate::core::{Texture2, Texture3};
use nalgebra::{Vector2, Vector3};
use palette::LinSrgb;

pub struct Color {
    color: LinSrgb<f64>,
}

impl Color {
    pub fn new(color: LinSrgb<f64>) -> Self {
        Self { color }
    }
}

impl Texture2 for Color {
    fn sample(&self, _uv: Vector2<f64>) -> LinSrgb<f64> {
        self.color
    }
}

impl Texture3 for Color {
    fn sample(&self, _uvw: Vector3<f64>) -> LinSrgb<f64> {
        self.color
    }
}
