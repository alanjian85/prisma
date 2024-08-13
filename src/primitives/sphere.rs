use encase::ShaderType;
use glam::Vec3;
use mlua::{prelude::*, UserData};

#[derive(FromLua, Clone, ShaderType)]
pub struct Sphere {
    center: Vec3,
    radius: f32,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }
}

impl UserData for Sphere {}
