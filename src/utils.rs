use nalgebra::{Vector2, Vector3};
use palette::Srgb;
use rand::prelude::*;

pub fn is_vec3_near_zero(vec: Vector3<f64>) -> bool {
    const THRESHOLD: f64 = 1e-8;
    vec.x.abs() < THRESHOLD && vec.y.abs() < THRESHOLD && vec.z.abs() < THRESHOLD
}

pub fn unit_vec3_to_rgb(vec: Vector3<f64>) -> Srgb<f64> {
    let vec = 0.5 * (vec + Vector3::new(1.0, 1.0, 1.0));
    Srgb::new(vec.x, vec.y, vec.z)
}

pub fn rand_square_vec2(rng: &mut ThreadRng) -> Vector2<f64> {
    Vector2::new(rng.gen_range(-0.5..=0.5), rng.gen_range(-0.5..=0.5))
}

pub fn rand_disk_vec2(rng: &mut ThreadRng) -> Vector2<f64> {
    let mut vec;
    loop {
        vec = Vector2::new(rng.gen_range(-1.0..=1.0), rng.gen_range(-1.0..=1.0));
        if vec.magnitude_squared() <= 1.0 {
            break;
        }
    }
    vec
}

pub fn rand_unit_vec3(rng: &mut ThreadRng) -> Vector3<f64> {
    let mut vec;
    loop {
        vec = Vector3::new(
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
            rng.gen_range(-1.0..=1.0),
        );
        if vec.magnitude_squared() <= 1.0 {
            break;
        }
    }
    vec.normalize()
}
