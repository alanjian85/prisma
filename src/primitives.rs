use encase::ShaderType;
use glam::Vec3;
use mlua::{prelude::*, UserData};

use crate::core::Aabb3;

#[derive(FromLua, Clone, ShaderType)]
pub struct Triangle {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    material: u32,
}

impl Triangle {
    pub fn new(p0: Vec3, p1: Vec3, p2: Vec3, material: u32) -> Self {
        Self {
            p0,
            p1,
            p2,
            material,
        }
    }

    pub fn aabb(&self) -> Aabb3 {
        Aabb3::new()
            .union_point(self.p0)
            .union_point(self.p1)
            .union_point(self.p2)
    }
}

impl UserData for Triangle {}
