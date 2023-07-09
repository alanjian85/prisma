// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>

#include "core/ray.hpp"
#include "core/utils.h"
#include "core/vector.hpp"
#include "film.hpp"

enum class CameraType {
    Persp,
    Ortho
};

class Camera {
public:
    PRISM_CPU Camera(int width, int height, CameraType type, Point3f o,
                     Vector3f d, Vector3f up = Vector3f(0, 1, 0), Real fov = 90)
                  : film(width, height), type(type), o(o), d(normalize(d)),
                    up(up), tanHalfFov(tan(Real(0.5) * fov))
    {
        assert(cross(d, up).length() != 0);
        assert(0 < fov && fov < 2 * pi);
    }

    PRISM_CPU static void *operator new(size_t count) {
        Camera *camera;
        cudaError_t status = cudaMallocManaged(&camera, sizeof(Camera));
        assert(status == cudaSuccess);
        return camera;
    }

    PRISM_CPU static void operator delete(void *ptr) {
        cudaError_t status = cudaFree(ptr);
        assert(status == cudaSuccess);
    }

    PRISM_CPU_GPU Ray generateRay(Point2f p) const;

    Film film;
    CameraType type;
    Point3f o;
    Vector3f up;

private:
    Vector3f d;
    Real tanHalfFov;
};
