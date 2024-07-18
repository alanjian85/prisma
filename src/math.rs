use nalgebra::Vector3;

pub fn reflect(incident: Vector3<f64>, normal: Vector3<f64>) -> Vector3<f64> {
    incident - 2.0 * incident.dot(&normal) * normal
}

pub fn refract(incident: Vector3<f64>, normal: Vector3<f64>, eta: f64) -> Vector3<f64> {
    let perp = eta * (incident - incident.dot(&normal) * normal);
    let para = -(1.0 - perp.magnitude_squared()).sqrt() * normal;
    perp + para
}

pub fn reflectance(cosine: f64, eta: f64) -> f64 {
    let r = (1.0 - eta) / (1.0 + eta);
    let r = r * r;
    r + (1.0 - r) * (1.0 - cosine).powi(5)
}
