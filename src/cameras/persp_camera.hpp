// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CAMERAS_PERSP_CAMERA_HPP
#define PRISM_CAMERAS_PERSP_CAMERA_HPP

#include <config/types.h>
#include <core/vector.hpp>

#include "camera.hpp"

class PerspCamera : public Camera {
public:
    PRISM_CPU_GPU PerspCamera(void *pixels, int width, int height)
                      : Camera(pixels, width, height) {}

    PRISM_CPU_GPU Ray generateRay(Point2f p) const override {
        Real tangent = tan(fov * Real(0.5));
        Vector3f right = normalize(cross(d, Vector3f(0, 1, 0))) * tangent;
        Vector3f up = cross(right, d);
        Ray r;
        r.o = o;
        r.d = normalize(d + (p.x * 2 - 1) * right + (p.y * 2 - 1) * up);
        return r;
    }

    Point3f o;
    Vector3f d;
    Real fov;
};

#endif // PRISM_CAMERAS_PERSP_CAMERA_HPP
