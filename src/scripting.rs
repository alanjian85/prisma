use std::{cell::RefCell, rc::Rc};

use mlua::prelude::*;

use crate::{config::Config, meshes::Meshes, scene::Scene, textures::Textures};

mod camera;
mod meshes;
mod scene;
mod textures;
mod utils;

pub struct Scripting {
    lua: Lua,
}

impl Scripting {
    pub fn new(textures: Rc<RefCell<Textures>>, meshes: Rc<RefCell<Meshes>>) -> LuaResult<Self> {
        let lua = Lua::new();

        let model_primitives = Rc::new(RefCell::new(Vec::new()));
        textures::init(&lua, textures)?;
        meshes::init(&lua, meshes, model_primitives.clone())?;

        let camera = lua.create_table()?;
        lua.globals().set("camera", camera)?;

        let scene = Scene::new(model_primitives.clone());
        lua.globals().set("scene", scene)?;

        Ok(Self { lua })
    }

    pub fn load(self, config: &Config, script: &str) -> LuaResult<Scene> {
        self.lua.load(script).exec()?;
        let camera = camera::load(&self.lua, config)?;
        let mut scene: Scene = self.lua.globals().get("scene")?;
        scene.set_camera(camera);
        Ok(scene)
    }
}
