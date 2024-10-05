use encase::ShaderType;

use crate::primitives::Primitives;

use super::Aabb3;

#[derive(ShaderType, Copy, Clone)]
pub struct Triangle {
    pub primitive: u32,
    pub v0: u32,
    pub v1: u32,
    pub v2: u32,
}

impl Triangle {
    pub fn aabb(&self, primitives: &Primitives) -> Aabb3 {
        Aabb3::new()
            .union_point(primitives.vertex(self.primitive, self.v0).pos)
            .union_point(primitives.vertex(self.primitive, self.v1).pos)
            .union_point(primitives.vertex(self.primitive, self.v2).pos)
    }
}
