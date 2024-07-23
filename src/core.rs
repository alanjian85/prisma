mod camera;
mod material;
mod primitive;
mod ray;
mod scene;
mod texture;

pub use camera::{Camera, CameraBuilder};
pub use material::Material;
pub use primitive::{Primitive, RayIntersection};
pub use ray::Ray;
pub use scene::Scene;
pub use texture::{Texture2, Texture3};
