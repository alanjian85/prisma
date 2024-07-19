mod camera;
mod material;
pub mod primitive;
mod ray;
mod scene;

pub use camera::{Camera, CameraBuilder};
pub use material::Material;
pub use primitive::{Primitive, RayIntersection};
pub use ray::Ray;
pub use scene::Scene;
