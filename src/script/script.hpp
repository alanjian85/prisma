// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#pragma once

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

class Script {
public:
    Script() {
        lua = luaL_newstate();
        luaL_openlibs(lua);
    }

    ~Script() {
        lua_close(lua);
    }

    bool load(const char *filename, const char *&message) {
        if (luaL_dofile(lua, filename)) {
            message = strdup(lua_tostring(lua, -1));
            lua_pop(lua, 1);
            return false;
        }
        return true;
    }

    std::unique_ptr<Camera> getCamera() const {
        return makeCamera(lua, "camera");
    }

private:
    Vector3f getLuaVector3f(lua_State *lua) {
        lua_rawgeti(lua, -1, 1);
        Real x = lua_tonumber(lua, -1);
        lua_rawgeti(lua, -2, 2);
        Real y = lua_tonumber(lua, -1);
        lua_rawgeti(lua, -3, 3);
        Real z = lua_tonumber(lua, -1);
        lua_pop(lua, 3);
        return Vector3f(x, y, z);
    }

    lua_State *lua;
};
