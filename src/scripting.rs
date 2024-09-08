use std::{cell::RefCell, rc::Rc};

use mlua::prelude::*;

use crate::{config::Config, models::Models, scene::Scene, textures::Textures};

mod camera;
mod models;
mod scene;
mod textures;
mod utils;

pub struct Scripting {
    lua: Lua,
}

impl Scripting {
    pub fn new(textures: Rc<RefCell<Textures>>, models: Rc<RefCell<Models>>) -> LuaResult<Self> {
        let lua = Lua::new();

        textures::init(&lua, textures)?;
        models::init(&lua, models.clone())?;

        let camera = lua.create_table()?;
        lua.globals().set("camera", camera)?;

        let scene = Scene::new(models);
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
