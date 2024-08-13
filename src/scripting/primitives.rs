use mlua::{prelude::*, Table};

use crate::primitives::Sphere;

use super::utils;

pub fn init(lua: &Lua) -> LuaResult<()> {
    let sphere = lua.create_table()?;
    sphere.set(
        "new",
        lua.create_function(move |_lua, (center, radius): (Table, f32)| {
            Ok(Sphere::new(utils::table_to_vec3(&center)?, radius))
        })?,
    )?;
    lua.globals().set("Sphere", sphere)?;
    Ok(())
}
