use encase::ShaderType;
use glam::Vec3;
use mlua::{prelude::*, UserData};

use crate::core::Aabb3;

#[derive(FromLua, Clone, ShaderType)]
pub struct Sphere {
    center: Vec3,
    radius: f32,
    materials: u32,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32, materials: u32) -> Self {
        Self {
            center,
            radius,
            materials,
        }
    }

    pub fn aabb(&self) -> Aabb3 {
        let r = Vec3::new(self.radius, self.radius, self.radius);
        Aabb3::from_corners(self.center + r, self.center - r)
    }
}

impl UserData for Sphere {}
