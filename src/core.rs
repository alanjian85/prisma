mod camera;
mod material;
mod primitive;
mod ray;
mod scene;

pub use camera::Camera;
pub use material::Material;
pub use primitive::{Primitive, RayIntersection};
pub use ray::Ray;
pub use scene::Scene;
