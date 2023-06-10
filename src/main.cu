// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include <config/types.h>
#include <core/film.hpp>
#include <core/utils.h>
#include <shapes/triangle.hpp>

const int tileSize = 16;

PRISM_KERNEL void render(prism::film film) {
    int nTilesX = (film.width + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    float u = static_cast<real_t>(x) / (film.width - 1);
    float v = static_cast<real_t>(y) / (film.height - 1);
    prism::triangle triangle(prism::point3f(-1, -1, 1),
                             prism::point3f( 1, -1, 1),
                             prism::point3f( 0,  1, 1));
    prism::ray ray(prism::point3f(u * 2 - 1, 1 - v * 2, 0),
                   prism::vector3f(0, 0, 1));
    if (triangle.intersect(ray)) {
        film.add_sample(prism::point2i(x, y), prism::color(u, v, 0.25));
    } else {
        film.add_sample(prism::point2i(x, y), prism::color(0, 0, 0));
    }
}

int main() {
    prism::film film(256, 256);
    int nTiles = ((film.width + tileSize - 1) / tileSize) *
                 ((film.height + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(film);
    cudaDeviceSynchronize();
    film.write_image("image.png");
    film.free();
    return 0;
}
