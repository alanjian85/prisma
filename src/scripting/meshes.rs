use std::{cell::RefCell, rc::Rc};

use mlua::prelude::*;

use crate::{core::Primitive, meshes::Meshes};

pub fn init(
    lua: &Lua,
    meshes: Rc<RefCell<Meshes>>,
    model_primitives: Rc<RefCell<Vec<Vec<Primitive>>>>,
) -> LuaResult<()> {
    {
        let meshes = meshes.clone();
        let model_primitives = model_primitives.clone();
        let model = lua.create_table()?;
        model.set(
            "load",
            lua.create_function(move |_lua, path: String| {
                let primitives = meshes
                    .borrow_mut()
                    .load_model(&path)
                    .map_err(|err| err.into_lua_err())?;
                let mut model_primitives = model_primitives.borrow_mut();
                let idx = model_primitives.len();
                model_primitives.push(primitives);
                Ok(idx)
            })?,
        )?;
        lua.globals().set("Model", model)?;
    }

    Ok(())
}
