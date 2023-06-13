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

    PRISM_CPU_GPU Ray generateRay(Point2i p) const override {
        Vector3f right = normalize(cross(d, Vector3f(0, 1, 0)));
        Vector3f up = cross(d, right);
        right *= focal * tan(fov * 0.5);
        up *= focal * tan(fov * 0.5);
        Real u = static_cast<Real>(p.x) / (film.width() - 1) * 2 - 1;
        Real v = 1 - static_cast<Real>(p.y) / (film.height() - 1) * 2;
        Ray r;
        r.o = o;
        r.d = normalize(focal * d + u * right + v * up);
        return r;
    }

    Point3f o;
    Vector3f d;
    Real focal, fov;
};

#endif // PRISM_CAMERAS_PERSP_CAMERA_HPP
