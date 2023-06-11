// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CAMERAS_PERSP_CAMERA_HPP
#define PRISM_CAMERAS_PERSP_CAMERA_HPP

#include <config/types.h>
#include <core/vector.hpp>

#include "camera.hpp"

namespace prism {
    class persp_camera : public camera {
    public:
        PRISM_CPU_GPU persp_camera(void *pixels, int width, int height)
                          : camera(pixels, width, height) {}

        PRISM_CPU_GPU ray generate_ray(point2i p) const override {
            vector3f right = normalize(cross(vector3f(0, 1, 0), d));
            vector3f up = cross(d, right);
            right *= focal * tan(fov * 0.5);
            up *= focal * tan(fov * 0.5);
            real_t u = static_cast<real_t>(p.x) / (film.width - 1) * 2 - 1;
            real_t v = 1 - static_cast<real_t>(p.y) / (film.height - 1) * 2;
            ray r;
            r.o = o;
            r.d = normalize(focal * d + u * right + v * up);
            return r;
        }

        point3f o;
        vector3f d;
        real_t focal, fov;
    };
}

#endif // PRISM_CAMERAS_PERSP_CAMERA_HPP
