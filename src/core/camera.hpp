// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CORE_CAMERA_HPP
#define PRISM_CORE_CAMERA_HPP

#include <config/types.h>

#include "film.hpp"
#include "ray.hpp"
#include "vector.hpp"

enum class CameraType {
    Persp,
    Ortho
};

class Camera {
public:
    PRISM_CPU Camera(int width, int height)
                  : film(width, height) {}

    PRISM_CPU static void *operator new(std::size_t count) {
        Camera *camera;
        cudaMallocManaged(&camera, sizeof(Camera));
        return camera;
    }

    PRISM_CPU static void operator delete(void *ptr) {
        cudaFree(ptr);
    }

    PRISM_CPU_GPU Ray generateRay(Point2f p) const {
        Ray r;
        Vector3f right = normalize(cross(d, Vector3f(0, 1, 0)));
        Vector3f up = cross(right, d);
        if (type == CameraType::Persp) {
            Real tangent = tan(fov * Real(0.5));
            right *= tangent;
            up *= tangent;
            r.o = o;
            r.d = normalize(d + (p.x * 2 - 1) * right + (p.y * 2 - 1) * up);
        }
        else {
            r.o = o + (p.x * 2 - 1) * right + (p.y * 2 - 1) * up;
            r.d = d;
        }
        return r;
    }

    Film film;
    CameraType type;
    Point3f o;
    Vector3f d;
    Real fov;
};

#endif // PRISM_CORE_CAMERA_HPP
