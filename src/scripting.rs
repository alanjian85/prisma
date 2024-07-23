use crate::config::Config;
use crate::core::{Camera, Scene};
use mlua::prelude::*;

mod camera;
mod materials;
mod primitives;
mod scene;
mod textures;
mod utils;

pub struct Scripting {
    lua: Lua,
}

impl Scripting {
    pub fn new() -> LuaResult<Self> {
        let lua = Lua::new();

        materials::init(&lua)?;
        primitives::init(&lua)?;
        textures::init(&lua)?;

        let scene = Scene::new();
        lua.globals().set("scene", scene)?;

        let camera = lua.create_table()?;
        lua.globals().set("camera", camera)?;

        Ok(Self { lua })
    }

    pub fn load(&self, config: &Config, script: &String) -> LuaResult<(Camera, Scene)> {
        self.lua.load(script).exec()?;

        let camera = camera::load(&self.lua, config)?;
        let scene = self.lua.globals().get("scene")?;

        Ok((camera, scene))
    }
}
