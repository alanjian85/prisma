use encase::ShaderType;
use glam::Vec3;

#[derive(ShaderType, Copy, Clone)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
}
