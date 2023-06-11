// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include <cameras/persp_camera.hpp>
#include <core/utils.h>
#include <shapes/sphere.hpp>

const int tileSize = 16;

PRISM_KERNEL void construct_objects(prism::persp_camera *camera,
                      void *pixels, int width, int height)
{
    new (camera) prism::persp_camera(pixels, width, height); 
    camera->o = prism::point3f(0, 0, 0);
    camera->d = prism::vector3f(0, 0, 1);
    camera->focal = 1;
    camera->fov = prism::radians(90);
}

PRISM_KERNEL void render(prism::camera &camera) {
    int nTilesX = (camera.film.width + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    prism::sphere sphere(prism::point3f(0, 0, 2), 0.5);
    prism::ray ray = camera.generate_ray(prism::point2i(x, y));
    prism::interaction interaction;
    if (sphere.intersect(ray, interaction)) {
        camera.film.add_sample(prism::point2i(x, y), prism::normal_to_color(interaction.n));
    } else {
        camera.film.add_sample(prism::point2i(x, y), prism::color(0, 0, 0));
    }
}

int main() {
    const int width = 1024, height = 1024;
    void *pixels;
    cudaMallocManaged(&pixels, width * height * 3);
    prism::persp_camera *camera;
    cudaMallocManaged(&camera, sizeof(prism::persp_camera));
    construct_objects<<<1, 1>>>(camera, pixels, width, height);
    cudaDeviceSynchronize();
    int nTiles = ((camera->film.width + tileSize - 1) / tileSize) *
                 ((camera->film.height + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(*camera);
    cudaDeviceSynchronize();
    camera->film.write_image("image.png");
    cudaFree(camera);
    cudaFree(pixels); 
    return 0;
}
