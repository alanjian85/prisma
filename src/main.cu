// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <iostream>
#include <memory>

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}
#include <tiny_obj_loader.h>

#include "camera/camera.hpp"
#include "core/utils.h"
#include "scene/scene.hpp"
const int tileSize = 16;

PRISM_KERNEL void render(Camera &camera, Scene &scene) {
    int nTilesX = (camera.film.width() + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    Real u = static_cast<Real>(x) / (camera.film.width() - 1);
    Real v = static_cast<Real>(y) / (camera.film.height() - 1);
    Ray ray = camera.generateRay(Point2f(u, 1 - v));
    Interaction interaction;
    if (!scene.intersect(ray, interaction)) {
        camera.film.addSample(Point2f(u, v), Color(0, 0, 0));
        return;
    }
    if (dot(ray.d, interaction.n) > 0)
        interaction.n = -interaction.n;
    Real attenuation = 1 / (camera.o - interaction.p).lengthSquared();
    Vector3f lightDir = normalize(camera.o - interaction.p);
    Vector3f halfwayDir = lightDir;
    Real diffuse = fmax(dot(interaction.n, lightDir), Real(0.0));
    Real specular = pow(fmax(dot(interaction.n, halfwayDir), Real(0.0)), 32);
    Color color = (diffuse * normalToColor(interaction.n) + specular * Color(1, 1, 1)) * attenuation;
    color = clamp(color, Color(0), Color(1));
    camera.film.addSample(Point2f(u, v), color);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: prism <script>.lua\n";
        return EXIT_FAILURE;
    }

    lua_State *lua = luaL_newstate();
    luaL_openlibs(lua);
    if (luaL_dofile(lua, argv[1])) {
        std::cerr << lua_tostring(lua, -1) << '\n';
        return EXIT_FAILURE;
    }

    tinyobj::ObjReader reader;
    reader.ParseFromFile("scene/viking_room.obj");

    std::vector<Triangle> primitives;
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            tinyobj::index_t idx1 = shapes[s].mesh.indices[index_offset];
            tinyobj::real_t x1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 0];
            tinyobj::real_t y1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 1];
            tinyobj::real_t z1 = attrib.vertices[3 * size_t(idx1.vertex_index) + 2];
            ++index_offset;

            tinyobj::index_t idx2 = shapes[s].mesh.indices[index_offset];
            tinyobj::real_t x2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 0];
            tinyobj::real_t y2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 1];
            tinyobj::real_t z2 = attrib.vertices[3 * size_t(idx2.vertex_index) + 2];
            ++index_offset;

            tinyobj::index_t idx3 = shapes[s].mesh.indices[index_offset];
            tinyobj::real_t x3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 0];
            tinyobj::real_t y3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 1];
            tinyobj::real_t z3 = attrib.vertices[3 * size_t(idx3.vertex_index) + 2];
            ++index_offset;

            primitives.push_back(Triangle(Vector3f(x1, y1, z1),
                                          Vector3f(x2, y2, z2),
                                          Vector3f(x3, y3, z3)));
        }
    }
    auto scene = std::make_unique<Scene>(primitives);
    auto camera = makeCamera(lua, "camera");

    int nTiles = ((camera->film.width() + tileSize - 1) / tileSize) *
                 ((camera->film.height() + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(*camera, *scene);
    cudaDeviceSynchronize();
    camera->film.writeImage("image.png");

    return EXIT_SUCCESS;
}
