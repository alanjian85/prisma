use std::{cell::RefCell, rc::Rc};

use mlua::{prelude::*, Table};

use crate::materials::Materials;

use super::utils;

pub fn init(lua: &Lua, materials: Rc<RefCell<Materials>>) -> LuaResult<()> {
    {
        let materials = materials.clone();
        let lambertian = lua.create_table()?;
        lambertian.set(
            "new",
            lua.create_function(move |_lua, albedo: Table| {
                Ok(materials
                    .borrow_mut()
                    .create_lambertian(utils::table_to_vec3(&albedo)?))
            })?,
        )?;
        lua.globals().set("Lambertian", lambertian)?;
    }

    {
        let materials = materials.clone();
        let metal = lua.create_table()?;
        metal.set(
            "new",
            lua.create_function(move |_lua, (albedo, fuzziness): (Table, f32)| {
                Ok(materials
                    .borrow_mut()
                    .create_metal(utils::table_to_vec3(&albedo)?, fuzziness))
            })?,
        )?;
        lua.globals().set("Metal", metal)?;
    }

    {
        let materials = materials.clone();
        let dielectric = lua.create_table()?;
        dielectric.set(
            "new",
            lua.create_function(move |_lua, ior: f32| {
                Ok(materials.borrow_mut().create_dielectric(ior))
            })?,
        )?;
        lua.globals().set("Dielectric", dielectric)?;
    }

    Ok(())
}
