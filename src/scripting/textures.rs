use std::{cell::RefCell, rc::Rc};

use mlua::prelude::*;

use crate::textures::Textures;

pub fn init(lua: &Lua, textures: Rc<RefCell<Textures>>) -> LuaResult<()> {
    let image_hdr = lua.create_table()?;
    image_hdr.set(
        "new",
        lua.create_function(move |_lua, path: String| {
            textures
                .borrow_mut()
                .create_image_hdr(&path)
                .map_err(|err| err.into_lua_err())
        })?,
    )?;
    lua.globals().set("ImageHdr", image_hdr)?;

    Ok(())
}
