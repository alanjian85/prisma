// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include <core/camera.hpp>
#include <core/scene.hpp>
#include <core/utils.h>

const int tileSize = 16;

PRISM_KERNEL void render(Camera &camera, Scene &scene) {
    int nTilesX = (camera.film.width() + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    Real u = static_cast<Real>(x) / (camera.film.width() - 1);
    Real v = static_cast<Real>(y) / (camera.film.height() - 1);
    Ray ray = camera.generateRay(Point2f(u, 1 - v));
    Interaction interaction;
    if (scene.intersect(ray, interaction)) {
        camera.film.addSample(Point2f(u, v), normalToColor(interaction.n));
    } else {
        camera.film.addSample(Point2f(u, v), Color(0, 0, 0));
    }
}

int main() {
    Scene *scene = new Scene();
    scene->addTriangle(Triangle(Vector3f(-1, -1, -2),
                                Vector3f( 1, -1, -2),
                                Vector3f( 0,  1, -2)));

    const int width = 1024, height = 1024;
    Camera *camera = new Camera(width, height);
    camera->type = CameraType::Ortho;
    camera->o = Vector3f(0, 0, 0);
    camera->d = Vector3f(0, 0, -1);
    camera->fov = radians(90);

    int nTiles = ((camera->film.width() + tileSize - 1) / tileSize) *
                 ((camera->film.height() + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(*camera, *scene);
    cudaDeviceSynchronize();
    camera->film.writeImage("image.png");

    delete scene;
    delete camera;
    return 0;
}
