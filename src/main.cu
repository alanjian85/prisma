// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include <cameras/persp_camera.hpp>
#include <config/types.h>
#include <core/film.hpp>
#include <core/utils.h>
#include <shapes/sphere.hpp>

const int tileSize = 16;

PRISM_KERNEL void render(prism::persp_camera &camera) {
    int nTilesX = (camera.film.width + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    prism::sphere sphere(prism::point3f(0, 0, 2), 0.5);
    prism::ray ray = camera.generate_ray(prism::point2i(x, y));
    if (sphere.intersect(ray)) {
        camera.film.add_sample(prism::point2i(x, y), prism::color(1, 1, 1));
    } else {
        camera.film.add_sample(prism::point2i(x, y), prism::color(0, 0, 0));
    }
}

int main() {
    prism::persp_camera *camera;
    cudaMallocManaged(&camera, sizeof(prism::persp_camera));
    new (camera) prism::persp_camera(256, 256);
    camera->o = prism::point3f(0, 0, 0);
    camera->d = prism::vector3f(0, 0, 1);
    camera->near = 1;
    camera->far = 1000;
    int nTiles = ((camera->film.width + tileSize - 1) / tileSize) *
                 ((camera->film.height + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(*camera);
    cudaDeviceSynchronize();
    camera->film.write_image("image.png");
    cudaFree(camera);
    return 0;
}
