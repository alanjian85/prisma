use mlua::{prelude::*, Table};
use nalgebra::{Point3, Vector3};
use palette::LinSrgb;

pub fn table_to_point3(table: &Table) -> LuaResult<Point3<f64>> {
    Ok(Point3::new(table.get(1)?, table.get(2)?, table.get(3)?))
}

pub fn table_to_vector3(table: &Table) -> LuaResult<Vector3<f64>> {
    Ok(Vector3::new(table.get(1)?, table.get(2)?, table.get(3)?))
}

pub fn table_to_color(table: &Table) -> LuaResult<LinSrgb<f64>> {
    Ok(LinSrgb::new(table.get(1)?, table.get(2)?, table.get(3)?))
}
