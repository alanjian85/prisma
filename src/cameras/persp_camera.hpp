// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CAMERAS_PERSP_CAMERA_HPP
#define PRISM_CAMERAS_PERSP_CAMERA_HPP

#include <config/types.h>
#include <core/point.hpp>
#include <core/ray.hpp>
#include <core/vector.hpp>

#include "camera.hpp"

namespace prism {
    class persp_camera : public camera {
    public:
        persp_camera(int width, int height)
            : camera(width, height) {}

        PRISM_CPU_GPU ray generate_ray(point2i p) const {
            vector3f right = normalize(cross(vector3f(0, 1, 0), d));
            vector3f up = cross(d, right);
            real_t u = static_cast<real_t>(p.x) / (film.width - 1) - 0.5;
            real_t v = 0.5 - static_cast<real_t>(p.y) / (film.height - 1);
            ray r;
            r.o = o;
            r.d = normalize(near * d + u * right + v * up);
            return r;
        }

        point3f o;
        vector3f d;
        real_t near, far;
    };
}

#endif // PRISM_CAMERAS_PERSP_CAMERA_HPP
