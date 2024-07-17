use nalgebra::Vector3;
use palette::Srgb;

pub fn unit_vec_to_rgb(vec: Vector3<f64>) -> Srgb<f64> {
    let vec = 0.5 * (vec + Vector3::new(1.0, 1.0, 1.0));
    Srgb::new(vec.x, vec.y, vec.z)
}
