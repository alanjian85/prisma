use mlua::prelude::*;

pub fn init(lua: &Lua) -> LuaResult<()> {
    //    triangle.set(
    //        "new",
    //        lua.create_function(
    //            move |_lua, (p0, p1, p2, material): (Table, Table, Table, u32)| {
    //                Ok(Triangle::new(
    //                    utils::table_to_vec3(&p0)?,
    //                    utils::table_to_vec3(&p1)?,
    //                    utils::table_to_vec3(&p2)?,
    //                    material,
    //                ))
    //            },
    //        )?,
    //    )?;
    //    lua.globals().set("Triangle", triangle)?;

    Ok(())
}
