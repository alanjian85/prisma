// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CAMERAS_ORTHO_CAMERA_HPP
#define PRISM_CAMERAS_ORTHO_CAMERA_HPP

#include <config/types.h>
#include <core/vector.hpp>

#include "camera.hpp"

class OrthoCamera : public Camera {
public:
    PRISM_CPU_GPU OrthoCamera(void *pixels, int width, int height)
                     : Camera(pixels, width, height) {}

    PRISM_CPU_GPU Ray generateRay(Point2f p) const override {
        Vector3f right = normalize(cross(d, Vector3f(0, 1, 0)));
        Vector3f up = cross(right, d);
        Ray r;
        r.o = o + (p.x * 2 - 1) * right + (p.y * 2 - 1) * up;
        r.d = d;
        return r;
    }

    Point3f o;
    Vector3f d;
};

#endif // PRISM_CAMERAS_ORTHO_CAMERA_HPP
