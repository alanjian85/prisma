use mlua::{prelude::*, UserData, UserDataMethods, Value};

use crate::core::Scene;

impl UserData for Scene {
    fn add_methods<'lua, M: UserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_method_mut("set_env_map", |_, this, env_map: u32| {
            this.set_env_map(env_map);
            Ok(())
        });
        //        methods.add_method_mut("add", |_, this, triangle: Triangle| {
        //            this.add(triangle);
        //            Ok(())
        //        });
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
