use std::{cell::RefCell, rc::Rc};

use mlua::prelude::*;

use crate::models::Models;

pub fn init(lua: &Lua, models: Rc<RefCell<Models>>) -> LuaResult<()> {
    {
        let models = models.clone();
        let model = lua.create_table()?;
        model.set(
            "load",
            lua.create_function(move |_lua, path: String| {
                models
                    .borrow_mut()
                    .load(&path)
                    .map_err(|err| err.into_lua_err())
            })?,
        )?;
        lua.globals().set("Model", model)?;
    }

    Ok(())
}
