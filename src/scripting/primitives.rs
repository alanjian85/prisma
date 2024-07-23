use crate::core::Primitive;
use crate::primitives::{Quad, Sphere};
use crate::scripting::materials::MaterialPtr;
use crate::scripting::utils;
use mlua::{prelude::*, Table, UserData};
use std::sync::Arc;

#[derive(FromLua, Clone)]
pub struct PrimitivePtr {
    pub ptr: Arc<dyn Primitive>,
}

impl UserData for PrimitivePtr {}

pub fn init(lua: &Lua) -> LuaResult<()> {
    let quad = lua.create_table()?;
    quad.set(
        "new",
        lua.create_function(
            |_lua, (orig, u, v, material): (Table, Table, Table, MaterialPtr)| {
                let quad = Quad::new(
                    utils::table_to_point3(&orig)?,
                    utils::table_to_vector3(&u)?,
                    utils::table_to_vector3(&v)?,
                    material.ptr,
                );
                Ok(PrimitivePtr {
                    ptr: Arc::new(quad),
                })
            },
        )?,
    )?;
    lua.globals().set("Quad", quad)?;

    let sphere = lua.create_table()?;
    sphere.set(
        "new",
        lua.create_function(
            |_lua, (center, radius, material): (Table, f64, MaterialPtr)| {
                let sphere = Sphere::new(utils::table_to_point3(&center)?, radius, material.ptr);
                Ok(PrimitivePtr {
                    ptr: Arc::new(sphere),
                })
            },
        )?,
    )?;
    lua.globals().set("Sphere", sphere)?;

    Ok(())
}
