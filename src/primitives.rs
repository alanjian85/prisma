use encase::ShaderType;
use mlua::{prelude::*, UserData};

use crate::{core::Aabb3, model::Vertex};

#[derive(FromLua, Clone, ShaderType)]
pub struct Triangle {
    pub p0: u32,
    pub p1: u32,
    pub p2: u32,
}

impl Triangle {
    pub fn aabb(&self, vertices: &[Vertex]) -> Aabb3 {
        Aabb3::new()
            .union_point(vertices[self.p0 as usize].pos)
            .union_point(vertices[self.p1 as usize].pos)
            .union_point(vertices[self.p2 as usize].pos)
    }
}

impl UserData for Triangle {}
