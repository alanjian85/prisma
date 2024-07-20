use crate::config::Config;
use crate::core::{Camera, CameraBuilder, Material, Primitive, Scene};
use crate::materials::{Dielectric, Lambertian, Metal};
use crate::primitives::Sphere;
use mlua::{prelude::*, Table, UserData, UserDataMethods, Value};
use nalgebra::{Point3, Vector3};
use palette::LinSrgb;
use std::sync::Arc;

pub struct Scripting {
    lua: Lua,
}

impl Scripting {
    pub fn new() -> LuaResult<Self> {
        let lua = Lua::new();

        Self::init_materials(&lua)?;
        Self::init_primitives(&lua)?;

        let scene = Scene::new();
        lua.globals().set("scene", scene)?;

        let camera = lua.create_table()?;
        lua.globals().set("camera", camera)?;

        Ok(Self { lua })
    }

    fn init_materials(lua: &Lua) -> LuaResult<()> {
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
            lua.create_function(|_lua, albedo: Table| {
                let lambertian = Lambertian::new(table_to_color(&albedo)?);
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
                let metal = Metal::new(table_to_color(&albedo)?, fuzziness);
                Ok(MaterialPtr {
                    ptr: Arc::new(metal),
                })
            })?,
        )?;
        lua.globals().set("Metal", metal)?;

        Ok(())
    }

    fn init_primitives(lua: &Lua) -> LuaResult<()> {
        let sphere = lua.create_table()?;
        sphere.set(
            "new",
            lua.create_function(
                |_lua, (center, radius, material): (Table, f64, MaterialPtr)| {
                    let sphere = Sphere::new(table_to_point3(&center)?, radius, material.ptr);
                    Ok(PrimitivePtr {
                        ptr: Arc::new(sphere),
                    })
                },
            )?,
        )?;
        lua.globals().set("Sphere", sphere)?;

        Ok(())
    }

    pub fn load(&self, config: &Config, script: &String) -> LuaResult<(Camera, Scene)> {
        self.lua.load(script).exec()?;

        let camera = self.load_camera(config)?;
        let scene = self.lua.globals().get("scene")?;

        Ok((camera, scene))
    }

    fn load_camera(&self, config: &Config) -> LuaResult<Camera> {
        let mut builder = CameraBuilder::new(config.size.width, config.size.height);
        let camera: Table = self.lua.globals().get("camera")?;

        if let Ok(pos) = camera.get("pos") {
            let pos = table_to_point3(&pos)?;
            builder.pos(pos);
        }
        if let Ok(center) = camera.get("center") {
            let center = table_to_point3(&center)?;
            builder.center(center);
        }
        if let Ok(up) = camera.get("up") {
            let up = table_to_vector3(&up)?;
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

        Ok(builder.build())
    }
}

#[derive(FromLua, Clone)]
struct MaterialPtr {
    ptr: Arc<dyn Material>,
}

impl UserData for MaterialPtr {}

#[derive(FromLua, Clone)]
struct PrimitivePtr {
    ptr: Arc<dyn Primitive>,
}

impl UserData for PrimitivePtr {}

impl UserData for Scene {
    fn add_methods<'lua, M: UserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_method_mut("add", |_, this, primitive: PrimitivePtr| {
            this.add(primitive.ptr);
            Ok(())
        });
    }
}

impl<'lua> FromLua<'lua> for Scene {
    fn from_lua(value: Value<'lua>, _lua: &'lua Lua) -> LuaResult<Self> {
        match value {
            Value::UserData(ud) => Ok(ud.take()?),
            _ => Err(mlua::Error::FromLuaConversionError {
                from: value.type_name(),
                to: "Scene",
                message: None,
            }),
        }
    }
}

fn table_to_point3(table: &Table) -> LuaResult<Point3<f64>> {
    Ok(Point3::new(table.get(1)?, table.get(2)?, table.get(3)?))
}

fn table_to_vector3(table: &Table) -> LuaResult<Vector3<f64>> {
    Ok(Vector3::new(table.get(1)?, table.get(2)?, table.get(3)?))
}

fn table_to_color(table: &Table) -> LuaResult<LinSrgb<f64>> {
    Ok(LinSrgb::new(table.get(1)?, table.get(2)?, table.get(3)?))
}
