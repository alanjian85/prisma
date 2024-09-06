use encase::ShaderType;
use glam::Vec3;
use tobj::{LoadError, LoadOptions};

use crate::primitives::Triangle;

#[derive(ShaderType)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
}

pub struct Model {
    models: Vec<tobj::Model>,
}

impl Model {
    pub fn load(filename: &str) -> Result<Self, LoadError> {
        let (models, _materials) = tobj::load_obj(
            filename,
            &LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )?;
        Ok(Self { models })
    }

    pub fn vertices(&self) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        for model in &self.models {
            let mesh = &model.mesh;
            for v in 0..mesh.positions.len() / 3 {
                vertices.push(Vertex {
                    pos: Vec3::new(
                        mesh.positions[3 * v],
                        mesh.positions[3 * v + 1],
                        mesh.positions[3 * v + 2],
                    ),
                    normal: Vec3::new(
                        mesh.normals[3 * v],
                        mesh.normals[3 * v + 1],
                        mesh.normals[3 * v + 2],
                    ),
                });
            }
        }
        vertices
    }

    pub fn primitives(&self, mut offset: u32) -> Vec<Triangle> {
        let mut primitives = Vec::new();
        for model in &self.models {
            let mesh = &model.mesh;
            for i in 0..mesh.indices.len() / 3 {
                let triangle = Triangle {
                    p0: mesh.indices[3 * i] + offset,
                    p1: mesh.indices[3 * i + 1] + offset,
                    p2: mesh.indices[3 * i + 2] + offset,
                };
                primitives.push(triangle);
            }
            offset += (mesh.positions.len() / 3) as u32;
        }
        primitives
    }
}
