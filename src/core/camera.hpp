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
    PRISM_CPU Camera(int width, int height, CameraType type, Point3f o,
                     Vector3f d, Vector3f up = Vector3f(0, 1, 0), Real fov = 90)
                  : film(width, height), type(type), o(o), d(normalize(d)),
                    up(up), fov(fov)
    {
        assert(cross(d, up).length() != 0);
    }

    PRISM_CPU static void *operator new(std::size_t count) {
        Camera *camera;
        cudaError_t status = cudaMallocManaged(&camera, sizeof(Camera));
        assert(status == cudaSuccess);
        return camera;
    }

    PRISM_CPU static void operator delete(void *ptr) {
        cudaError_t status = cudaFree(ptr);
        assert(status == cudaSuccess);
    }

    PRISM_CPU_GPU Ray generateRay(Point2f p) const {
        Ray r;
        Vector3f right = normalize(cross(d, up));
        Vector3f newUp = cross(right, d);
        if (type == CameraType::Persp) {
            assert(0 < fov && fov < 2 * pi);
            Real tangent = tan(fov * Real(0.5));
            right *= tangent;
            newUp *= tangent;
            r.o = o;
            r.d = normalize(d + (p.x * 2 - 1) * right + (p.y * 2 - 1) * newUp);
        }
        else {
            r.o = o + (p.x * 2 - 1) * right + (p.y * 2 - 1) * newUp;
            r.d = d;
        }
        return r;
    }

    Film film;

private:
    CameraType type;
    Point3f o;
    Vector3f d;
    Vector3f up;
    Real fov;
};

#endif // PRISM_CORE_CAMERA_HPP
