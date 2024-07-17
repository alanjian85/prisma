use nalgebra::Vector3;
use palette::Srgb;
use rand::prelude::*;

pub fn unit_vec_to_rgb(vec: Vector3<f64>) -> Srgb<f64> {
    let vec = 0.5 * (vec + Vector3::new(1.0, 1.0, 1.0));
    Srgb::new(vec.x, vec.y, vec.z)
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
