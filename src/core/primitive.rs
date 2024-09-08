use encase::ShaderType;
use mlua::{FromLua, UserData};

use crate::models::Meshes;

use super::Aabb3;

#[derive(FromLua, Clone, ShaderType)]
pub struct Primitive {
    pub idx: u32,
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
}

impl Primitive {
    pub fn aabb(&self, meshes: &Meshes) -> Aabb3 {
        Aabb3::new()
            .union_point(meshes.vertex(self.idx, self.v0).pos)
            .union_point(meshes.vertex(self.idx, self.v1).pos)
            .union_point(meshes.vertex(self.idx, self.v2).pos)
    }
}

impl UserData for Primitive {}
