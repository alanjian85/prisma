use nalgebra::{Vector2, Vector3};
use palette::Srgb;
use rand::{rngs::ThreadRng, Rng};

pub fn unit_vec3_to_rgb(vec: Vector3<f64>) -> Srgb<f64> {
    let vec = 0.5 * (vec + Vector3::new(1.0, 1.0, 1.0));
    Srgb::new(vec.x, vec.y, vec.z)
}

pub fn rand_square_vec2(rng: &mut ThreadRng) -> Vector2<f64> {
    Vector2::new(rng.gen_range(-0.5..=0.5), rng.gen_range(-0.5..=0.5))
}

pub fn rand_unit_vec3(rng: &mut ThreadRng) -> Vector3<f64> {
    let mut vec;
    loop {
        vec = Vector3::new(
            rng.gen_range(-0.5..=0.5),
            rng.gen_range(-0.5..=0.5),
            rng.gen_range(-0.5..=0.5),
        );
        if vec.magnitude_squared() <= 1.0 {
            break;
        }
    }
    vec.normalize()
}
