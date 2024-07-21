use crate::core::Scene;
use crate::scripting::primitives::PrimitivePtr;
use mlua::{prelude::*, UserData, UserDataMethods, Value};

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
