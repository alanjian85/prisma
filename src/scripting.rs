use std::{cell::RefCell, rc::Rc};

use mlua::prelude::*;

use crate::{core::Scene, textures::Textures};

mod scene;
mod textures;

pub struct Scripting {
    lua: Lua,
}

impl Scripting {
    pub fn new(textures: Rc<RefCell<Textures>>) -> LuaResult<Self> {
        let lua = Lua::new();

        textures::init(&lua, textures)?;

        let scene = Scene::new();
        lua.globals().set("scene", scene)?;

        Ok(Self { lua })
    }

    pub fn load(self, script: &str) -> LuaResult<Scene> {
        self.lua.load(script).exec()?;
        let scene = self.lua.globals().get("scene")?;
        Ok(scene)
    }
}
