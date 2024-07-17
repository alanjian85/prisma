use nalgebra::Vector3;
use palette::LinSrgb;

pub fn unit_vec_to_rgb(vec: Vector3<f64>) -> LinSrgb<f64> {
    let vec = 0.5 * (vec + Vector3::new(1.0, 1.0, 1.0));
    LinSrgb::new(vec.x, vec.y, vec.z)
}
