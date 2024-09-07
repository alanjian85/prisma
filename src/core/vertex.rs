use encase::ShaderType;
use glam::Vec3;

#[derive(ShaderType)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
}
