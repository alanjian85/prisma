use crate::core::Material;
use crate::materials::{Dielectric, Lambertian, Metal};
use crate::scripting::textures::Texture2Ptr;
use crate::scripting::utils;
use mlua::{prelude::*, Table, UserData};
use std::sync::Arc;

#[derive(FromLua, Clone)]
pub struct MaterialPtr {
    pub ptr: Arc<dyn Material>,
}

impl UserData for MaterialPtr {}

pub fn init(lua: &Lua) -> LuaResult<()> {
    let dielectric = lua.create_table()?;
    dielectric.set(
        "new",
        lua.create_function(|_lua, eta: f64| {
            let dielectric = Dielectric::new(eta);
            Ok(MaterialPtr {
                ptr: Arc::new(dielectric),
            })
        })?,
    )?;
    lua.globals().set("Dielectric", dielectric)?;

    let lambertian = lua.create_table()?;
    lambertian.set(
        "new",
        lua.create_function(|_lua, texture: Texture2Ptr| {
            let lambertian = Lambertian::new(texture.ptr);
            Ok(MaterialPtr {
                ptr: Arc::new(lambertian),
            })
        })?,
    )?;
    lua.globals().set("Lambertian", lambertian)?;

    let metal = lua.create_table()?;
    metal.set(
        "new",
        lua.create_function(|_lua, (albedo, fuzziness): (Table, f64)| {
            let metal = Metal::new(utils::table_to_color(&albedo)?, fuzziness);
            Ok(MaterialPtr {
                ptr: Arc::new(metal),
            })
        })?,
    )?;
    lua.globals().set("Metal", metal)?;

    Ok(())
}
