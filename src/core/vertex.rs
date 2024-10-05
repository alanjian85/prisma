use encase::ShaderType;
use glam::{Vec2, Vec3};

#[derive(ShaderType, Copy, Clone)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub tex_coord: Vec2,
}
