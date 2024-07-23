use crate::core::{Texture2, Texture3};
use crate::scripting::utils;
use crate::textures::{Color, Image, ImageHdr, Panorama};
use mlua::{prelude::*, Table, UserData};
use std::sync::Arc;

#[derive(FromLua, Clone)]
pub struct Texture2Ptr {
    pub ptr: Arc<dyn Texture2>,
}

impl UserData for Texture2Ptr {}

#[derive(FromLua, Clone)]
pub struct Texture3Ptr {
    pub ptr: Arc<dyn Texture3>,
}

impl UserData for Texture3Ptr {}

pub fn init(lua: &Lua) -> LuaResult<()> {
    let color2 = lua.create_table()?;
    color2.set(
        "new",
        lua.create_function(|_lua, color: Table| {
            let color = Color::new(utils::table_to_color(&color)?);
            Ok(Texture2Ptr {
                ptr: Arc::new(color),
            })
        })?,
    )?;
    lua.globals().set("Color2", color2)?;

    let color3 = lua.create_table()?;
    color3.set(
        "new",
        lua.create_function(|_lua, color: Table| {
            let color = Color::new(utils::table_to_color(&color)?);
            Ok(Texture3Ptr {
                ptr: Arc::new(color),
            })
        })?,
    )?;
    lua.globals().set("Color3", color3)?;

    let image = lua.create_table()?;
    image.set(
        "new",
        lua.create_function(|_lua, path: String| {
            let image = Image::new(&path).map_err(|err| err.into_lua_err())?;
            Ok(Texture2Ptr {
                ptr: Arc::new(image),
            })
        })?,
    )?;
    lua.globals().set("Image", image)?;

    let image_hdr = lua.create_table()?;
    image_hdr.set(
        "new",
        lua.create_function(|_lua, path: String| {
            let image_hdr = ImageHdr::new(&path).map_err(|err| err.into_lua_err())?;
            Ok(Texture2Ptr {
                ptr: Arc::new(image_hdr),
            })
        })?,
    )?;
    lua.globals().set("ImageHdr", image_hdr)?;

    let panorama = lua.create_table()?;
    panorama.set(
        "new",
        lua.create_function(|_lua, path: String| {
            let panorama = Panorama::new(&path).map_err(|err| err.into_lua_err())?;
            Ok(Texture3Ptr {
                ptr: Arc::new(panorama),
            })
        })?,
    )?;
    lua.globals().set("Panorama", panorama)?;

    Ok(())
}
