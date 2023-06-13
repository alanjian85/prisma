// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

//#include <cameras/persp_camera.hpp>
#include <cameras/ortho_camera.hpp>
#include <core/utils.h>
#include <shapes/triangle.hpp>
//#include <shapes/sphere.hpp>

const int tileSize = 16;
/*
PRISM_KERNEL void constructObjects(PerspCamera *camera,
                      void *pixels, int width, int height)
{
    new (camera) PerspCamera(pixels, width, height);
    camera->o = Point3f(0, 0, 0);
    camera->d = Vector3f(0, 0, -1);
    camera->fov = radians(90);
}
*/

PRISM_KERNEL void constructObjects(OrthoCamera *camera,
                      void *pixels, int width, int height)
{
    new (camera) OrthoCamera(pixels, width, height);
    camera->o = Point3f(0, 0, 0);
    camera->d = Vector3f(0, 0, -1);
}

PRISM_KERNEL void render(Camera &camera) {
    int nTilesX = (camera.film.width() + tileSize - 1) / tileSize;
    int x = blockIdx.x % nTilesX * tileSize + threadIdx.x % tileSize;
    int y = blockIdx.x / nTilesX * tileSize + threadIdx.x / tileSize;
    //Sphere sphere(Point3f(0, 0, -1), 0.5);
    Triangle triangle(Point3f(-1, -1, -1),
                      Point3f( 1, -1, -1),
                      Point3f( 0,  1, -1));
    Ray ray = camera.generateRay(Point2i(x, y));
    Interaction interaction;
    if (triangle.intersect(ray, interaction)) {
        camera.film.addSample(Point2i(x, y), normalToColor(interaction.n));
    } else {
        camera.film.addSample(Point2i(x, y), Color(0, 0, 0));
    }
}

int main() {
    const int width = 1024, height = 1024;
    void *pixels;
    cudaMallocManaged(&pixels, width * height * 3);
//    PerspCamera *camera;
//    cudaMallocManaged(&camera, sizeof(PerspCamera));
    OrthoCamera *camera;
    cudaMallocManaged(&camera, sizeof(OrthoCamera));
    constructObjects<<<1, 1>>>(camera, pixels, width, height);
    cudaDeviceSynchronize();
    int nTiles = ((camera->film.width() + tileSize - 1) / tileSize) *
                 ((camera->film.height() + tileSize - 1) / tileSize);
    render<<<nTiles, tileSize * tileSize>>>(*camera);
    cudaDeviceSynchronize();
    camera->film.writeImage("image.png");
    cudaFree(camera);
    cudaFree(pixels); 
    return 0;
}
