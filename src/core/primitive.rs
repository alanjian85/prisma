use std::{cell::RefCell, rc::Rc};

use encase::ShaderType;
use mlua::{FromLua, UserData};

use crate::meshes::Meshes;

use super::Aabb3;

#[derive(FromLua, Clone, ShaderType)]
pub struct Primitive {
    pub idx: u32,
    pub p0: u32,
    pub p1: u32,
    pub p2: u32,
}

impl Primitive {
    pub fn aabb(&self, meshes: &Rc<RefCell<Meshes>>) -> Aabb3 {
        let meshes = meshes.borrow();
        let offset = meshes.offsets[self.idx as usize] as usize;
        Aabb3::new()
            .union_point(meshes.vertices[self.p0 as usize + offset].pos)
            .union_point(meshes.vertices[self.p1 as usize + offset].pos)
            .union_point(meshes.vertices[self.p2 as usize + offset].pos)
    }
}

impl UserData for Primitive {}
