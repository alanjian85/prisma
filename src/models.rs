use tobj::{LoadError, GPU_LOAD_OPTIONS};

use crate::core::Primitive;

mod materials;
mod meshes;

pub use materials::Materials;
pub use meshes::Meshes;

#[derive(Default)]
pub struct Models {
    materials: Materials,
    meshes: Meshes,
    primitives: Vec<Vec<Primitive>>,
}

impl Models {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn materials(&self) -> &Materials {
        &self.materials
    }

    pub fn meshes(&self) -> &Meshes {
        &self.meshes
    }

    pub fn primitives(&self, idx: u32) -> &Vec<Primitive> {
        &self.primitives[idx as usize]
    }

    pub fn load(&mut self, path: &str) -> Result<u32, LoadError> {
        let (models, materials) = tobj::load_obj(path, &GPU_LOAD_OPTIONS)?;
        let materials = materials?;

        let material_idx_start = self.materials.len() as u32;
        for material in materials {
            self.materials.add(&material);
        }

        let mut primitives = Vec::new();
        for model in models {
            primitives.append(&mut self.meshes.add(&model.mesh, material_idx_start).clone());
        }

        self.primitives.push(primitives);
        Ok(self.primitives.len() as u32 - 1)
    }
}
