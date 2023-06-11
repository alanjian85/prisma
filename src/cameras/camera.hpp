// Copyright (C) 2023 Alan Jian (alanjian85@outlook.com)
// SPDX-License-Identifier: MIT

#ifndef PRISM_CAMERAS_CAMERA_HPP
#define PRISM_CAMERAS_CAMERA_HPP

#include <core/film.hpp>
#include <core/ray.hpp>
#include <core/utils.h>

namespace prism {
    class camera {
    public:
        PRISM_GPU camera(void *pixels, int width, int height)
                      : film(pixels, width, height) {}

        PRISM_CPU_GPU virtual ray generate_ray(point2i p) const = 0;

        prism::film film;
    };
}

#endif // PRISM_CAMERAS_CAMERA_HPP
