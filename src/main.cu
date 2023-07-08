// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include <iostream>
#include <memory>

#include <tiny_obj_loader.h>

#include "core/camera.hpp"
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
    scene.intersect(ray, interaction);
    if (dot(ray.d, interaction.n) > 0)
        interaction.n = -interaction.n;
    Real attenuation = 1 / (camera.o - interaction.p).lengthSquared();
    Vector3f lightDir = normalize(camera.o - interaction.p);
    Vector3f halfwayDir = lightDir;
    Real diffuse = dot(interaction.n, lightDir);
    Real specular = pow(dot(interaction.n, halfwayDir), 32);
    Color color = (diffuse * normalToColor(interaction.n) + specular * Color(1, 1, 1)) * attenuation;
    color = clamp(color, Color(0), Color(1));
    camera.film.addSample(Point2f(u, v), color);
}

int main() {
    tinyobj::ObjReader reader;
    reader.ParseFromFile("viking_room.obj");

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

    const int width = 1024, height = 1024;
    auto camera = std::make_unique<Camera>(width, height, CameraType::Persp,
                      Vector3f(1, 0, 1), Vector3f(-1, 0, -1), Vector3f(0, 0, 1),
                      radians(90));

    int nTiles = ((camera->film.width() + tileSize - 1) / tileSize) *
                 ((camera->film.height() + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(*camera, *scene);
    cudaDeviceSynchronize();
    camera->film.writeImage("image.png");

    return 0;
}
