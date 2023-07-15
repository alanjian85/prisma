// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include "camera.hpp"

PRISM_CPU_GPU Ray Camera::generateRay(Point2f p) const {
    Ray r;
    Vector3f right = normalize(cross(d, up));
    Vector3f newUp = cross(right, d);
    if (type == CameraType::Persp) {
        right *= tanHalfFov;
        newUp *= tanHalfFov;
        r.o = o;
        r.d = normalize(d + (p.x * 2 - 1) * right + (p.y * 2 - 1) * newUp);
    }
    else {
        r.o = o + (p.x * 2 - 1) * right + (p.y * 2 - 1) * newUp;
        r.d = d;
    }
    return r;
}

PRISM_CPU std::unique_ptr<Camera> makeCamera(lua_State *lua, const char *name) {
    lua_getglobal(lua, name);
    if (!lua_istable(lua, -1)) {
        lua_pop(lua, 1);
        return nullptr;
    }

    lua_getfield(lua, -1, "width");
    int width = lua_tointeger(lua, -1);
    lua_getfield(lua, -2, "height");
    int height = lua_tointeger(lua, -1);

    lua_getfield(lua, -3, "type");
    const char *typeStr = lua_tostring(lua, -1);
    CameraType type;
    if (strcmp(typeStr, "Persp") == 0) {
        type = CameraType::Persp;
    }
    else if (strcmp(typeStr, "Ortho") == 0) {
        type = CameraType::Ortho;
    }
    else {
        lua_pop(lua, 4);
        return nullptr;
    }

    lua_getfield(lua, -4, "o");
    Vector3f o = getLuaVector3f(lua);
    lua_getfield(lua, -5, "d");
    Vector3f d = getLuaVector3f(lua);
    Vector3f up(0, 1, 0);
    lua_getfield(lua, -6, "up");
    if (lua_istable(lua, -1))
        up = getLuaVector3f(lua);

    Real fov = 0.5 * pi;
    lua_getfield(lua, -7, "fov");
    if (lua_isnumber(lua, -1))
        fov = radians(lua_tonumber(lua, -1));

    lua_pop(lua, 8);
    return std::make_unique<Camera>(width, height, type, o, d, up, fov);
}
