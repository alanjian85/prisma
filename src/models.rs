use glam::Vec3;
use tobj::{LoadError, LoadOptions};

use crate::core::{Primitive, Vertex};

mod meshes;

pub use meshes::Meshes;

#[derive(Default)]
pub struct Models {
    meshes: Meshes,
    primitives: Vec<Vec<Primitive>>,
}

impl Models {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn meshes(&self) -> &Meshes {
        &self.meshes
    }

    pub fn primitives(&self, idx: u32) -> &Vec<Primitive> {
        &self.primitives[idx as usize]
    }

    pub fn load(&mut self, path: &str) -> Result<u32, LoadError> {
        let (models, _materials) = tobj::load_obj(
            path,
            &LoadOptions {
                single_index: true,
                triangulate: true,
                ..Default::default()
            },
        )?;

        let mut primitives = Vec::new();
        for model in models {
            let mesh = model.mesh;

            let mut vertices = Vec::with_capacity(mesh.positions.len() / 3);
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

            let idx = self.meshes.add_mesh(&mut vertices);

            for i in 0..mesh.indices.len() / 3 {
                primitives.push(Primitive {
                    idx,
                    v0: mesh.indices[3 * i],
                    v1: mesh.indices[3 * i + 1],
                    v2: mesh.indices[3 * i + 2],
                });
            }
        }

        self.primitives.push(primitives);
        Ok(self.primitives.len() as u32 - 1)
    }
}
