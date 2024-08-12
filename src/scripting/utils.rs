use glam::Vec3;
use mlua::{prelude::*, Table};

pub fn table_to_vec3(table: &Table) -> LuaResult<Vec3> {
    Ok(Vec3::new(table.get(1)?, table.get(2)?, table.get(3)?))
}
