use crate::core::Material;
use crate::materials::{Dielectric, Lambertian, Light, Metal};
use crate::scripting::textures::Texture2Ptr;
use mlua::{prelude::*, UserData};
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

    let light = lua.create_table()?;
    light.set(
        "new",
        lua.create_function(|_lua, texture: Texture2Ptr| {
            let light = Light::new(texture.ptr);
            Ok(MaterialPtr {
                ptr: Arc::new(light),
            })
        })?,
    )?;
    lua.globals().set("Light", light)?;

    let metal = lua.create_table()?;
    metal.set(
        "new",
        lua.create_function(|_lua, (texture, fuzziness): (Texture2Ptr, f64)| {
            let metal = Metal::new(texture.ptr, fuzziness);
            Ok(MaterialPtr {
                ptr: Arc::new(metal),
            })
        })?,
    )?;
    lua.globals().set("Metal", metal)?;

    Ok(())
}
