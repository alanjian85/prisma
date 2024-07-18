use nalgebra::Vector3;

pub fn reflect(incident: Vector3<f64>, normal: Vector3<f64>) -> Vector3<f64> {
    incident - 2.0 * incident.dot(&normal) * normal
}
