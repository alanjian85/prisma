use mlua::{prelude::*, Table};

use crate::{
    config::Config,
    scene::{Camera, CameraBuilder},
};

use super::utils;

pub fn load(lua: &Lua, config: &Config) -> LuaResult<Camera> {
    let mut builder = CameraBuilder::new();
    let camera: Table = lua.globals().get("camera")?;

    if let Ok(pos) = camera.get("pos") {
        let pos = utils::table_to_vec3(&pos)?;
        builder.pos(pos);
    }
    if let Ok(center) = camera.get("center") {
        let center = utils::table_to_vec3(&center)?;
        builder.center(center);
    }
    if let Ok(up) = camera.get("up") {
        let up = utils::table_to_vec3(&up)?;
        builder.up(up);
    }
    if let Ok(fov) = camera.get("fov") {
        builder.fov(fov);
    }
    if let Ok(focus_dist) = camera.get("focus_dist") {
        builder.focus_dist(focus_dist);
    }
    if let Ok(lens_angle) = camera.get("lens_angle") {
        builder.lens_angle(lens_angle);
    }

    Ok(builder.build(config.size.width, config.size.height))
}
