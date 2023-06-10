// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#include <config/types.h>
#include <core/film.hpp>

const int tileSize = 16;

__global__ void render(prism::film film) {
    int nTilesX = (film.width + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    float u = static_cast<real_t>(x) / (film.width - 1);
    float v = static_cast<real_t>(y) / (film.height - 1);
    film.add_sample(prism::point2i(x, y), prism::color(u, v, 0.25f));
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
