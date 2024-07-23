use crate::core::Scene;
use crate::scripting::primitives::PrimitivePtr;
use crate::scripting::textures::Texture3Ptr;
use mlua::{prelude::*, UserData, UserDataMethods, Value};

impl UserData for Scene {
    fn add_methods<'lua, M: UserDataMethods<'lua, Self>>(methods: &mut M) {
        methods.add_method_mut("add", |_, this, primitive: PrimitivePtr| {
            this.add(primitive.ptr);
            Ok(())
        });

        methods.add_method_mut("set_env", |_, this, env_map: Texture3Ptr| {
            this.set_env(env_map.ptr);
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
